import { useRef, useState, useCallback } from 'react';
import { AudioMessage } from '../types/order';

// WebRTC for microphone input, WebSocket + AudioWorklet for TTS output
export function useWebRTCAudioStream() {
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<string[]>([]);
  const [transcription, setTranscription] = useState<string>('');
  const [isTTSPlaying, setIsTTSPlaying] = useState(false);
  
  // WebRTC refs for microphone input
  const pc = useRef<RTCPeerConnection | null>(null);
  const localStream = useRef<MediaStream | null>(null);
  
  // WebSocket refs for TTS output
  const socket = useRef<WebSocket | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const ttsWorkletNode = useRef<AudioWorkletNode | null>(null);
  
  // Track recording state in ref to avoid stale closures
  const isRecordingRef = useRef(false);

  const handleAudioMessage = (message: AudioMessage) => {
    switch (message.type) {
      case 'transcription':
        if (message.text) {
          setTranscription(message.text);
          setMessages(prev => [...prev, `You said: "${message.text}"`]);
        }
        break;
      case 'audio_response':
        if (message.text) {
          setMessages(prev => [...prev, `AI: ${message.text}`]);
        }
        break;
      case 'error':
        setMessages(prev => [...prev, `Error: ${message.message}`]);
        break;
    }
  };

  // Convert base64 to Int16Array (from RealtimeVoiceChat)
  const base64ToInt16Array = (b64: string): Int16Array => {
    const raw = atob(b64);
    const buf = new ArrayBuffer(raw.length);
    const view = new Uint8Array(buf);
    for (let i = 0; i < raw.length; i++) {
      view[i] = raw.charCodeAt(i);
    }
    return new Int16Array(buf);
  };

  // Handle WebSocket messages for TTS audio
  const handleWebSocketMessage = (evt: MessageEvent) => {
    if (typeof evt.data === "string") {
      try {
        const msg = JSON.parse(evt.data);
        
        if (msg.type === "tts_chunk") {
          const int16Data = base64ToInt16Array(msg.content);
          
          if (ttsWorkletNode.current) {
            // Send audio data with timestamp for first chunk tracking
            ttsWorkletNode.current.port.postMessage({
              audioData: int16Data,
              timestamp: Date.now()
            });
          } else {
            console.warn('AudioWorklet not ready, dropping audio chunk');
          }
        } else if (msg.type === "tts_interruption") {
          if (ttsWorkletNode.current) {
            ttsWorkletNode.current.port.postMessage({ type: "clear" });
          }
          setIsTTSPlaying(false);
        } else if (msg.type === "stop_tts") {
          if (ttsWorkletNode.current) {
            ttsWorkletNode.current.port.postMessage({ type: "clear" });
          }
          setIsTTSPlaying(false);
        }
      } catch (e) {
        console.error("Error parsing WebSocket message:", e);
      }
    }
  };

  // Setup TTS AudioWorklet (from RealtimeVoiceChat)
  const setupTTSPlayback = async () => {
    if (!audioContext.current) {
      audioContext.current = new AudioContext({
        latencyHint: 0.01,  // Ultra-low latency
        sampleRate: 48000    // Match WebRTC sample rate
      });
    }

    try {
      await audioContext.current.audioWorklet.addModule('/ttsPlaybackProcessor.js');
      
      ttsWorkletNode.current = new AudioWorkletNode(
        audioContext.current,
        'tts-playback-processor'
      );

      ttsWorkletNode.current.port.onmessage = (event) => {
        const { type } = event.data;
        if (type === 'ttsPlaybackStarted') {
          if (!isTTSPlaying && socket.current && socket.current.readyState === WebSocket.OPEN) {
            setIsTTSPlaying(true);
            socket.current.send(JSON.stringify({ type: 'tts_start' }));
          }
        } else if (type === 'ttsPlaybackStopped') {
          if (isTTSPlaying && socket.current && socket.current.readyState === WebSocket.OPEN) {
            setIsTTSPlaying(false);
            socket.current.send(JSON.stringify({ type: 'tts_stop' }));
          }
        }
      };
      
      ttsWorkletNode.current.connect(audioContext.current.destination);
    } catch (error) {
      console.error('Failed to setup TTS AudioWorklet:', error);
    }
  };

  // Create WebRTC peer connection for microphone input
  const createPeerConnection = useCallback(() => {
    const peerConnection = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' }
      ]
    });

    // Handle incoming tracks (microphone echo - we'll ignore this)
    peerConnection.ontrack = (event) => {
      // Mute any WebRTC audio playback for debugging
      if (event.track.kind === 'audio') {
        const audio = new Audio();
        audio.srcObject = new MediaStream([event.track]);
        audio.autoplay = false; // Do not autoplay
        audio.volume = 0; // Mute
      }
    };

    return peerConnection;
  }, []);

  /**
   * Start - WebRTC for microphone, WebSocket for TTS
   */
  const start = useCallback(async (): Promise<void> => {
    console.log('Starting WebRTC + WebSocket connection...');
    setMessages(prev => [...prev, 'Starting WebRTC + WebSocket connection...']);

    // Setup WebSocket for TTS audio
    const wsUrl = 'ws://localhost:8002/api/ws';
    socket.current = new WebSocket(wsUrl);

    socket.current.onopen = async () => {
      setMessages(prev => [...prev, 'WebSocket: Connected for TTS']);
      await setupTTSPlayback();
    };

    socket.current.onmessage = handleWebSocketMessage;

    socket.current.onclose = (event) => {
      setMessages(prev => [...prev, `WebSocket: Closed (${event.code})`]);
    };

    socket.current.onerror = (err) => {
      console.error('WebSocket connection error:', err);
      setMessages(prev => [...prev, 'WebSocket: Error']);
    };

    // Create WebRTC peer connection for microphone
    pc.current = createPeerConnection();

    // Get user media - audio only with high quality settings
    const constraints: MediaStreamConstraints = {
      audio: {
        sampleRate: 48000, // Use WebRTC standard rate for better quality
        channelCount: 1, // Mono audio
        echoCancellation: false,    // Turn off echo cancellation
        noiseSuppression: false,    // Turn off noise suppression
        autoGainControl: false      // Turn off auto gain control
      },
      video: false
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      localStream.current = stream;
      
      setMessages(prev => [...prev, 'Microphone access granted']);

      // Add microphone track to WebRTC
      const audioTrack = stream.getAudioTracks()[0];
      if (audioTrack) {
        pc.current.addTrack(audioTrack, stream);
        setMessages(prev => [...prev, 'Microphone track added to WebRTC']);
      }

      // Create offer and send to backend
      const offer = await pc.current.createOffer();
      await pc.current.setLocalDescription(offer);

      const response = await fetch('http://localhost:8002/api/webrtc/offer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sdp: pc.current.localDescription?.sdp,
          type: pc.current.localDescription?.type,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const { sdp: answerSdp, type: answerType } = await response.json();
      const answer = new RTCSessionDescription({ sdp: answerSdp, type: answerType });
      await pc.current.setRemoteDescription(answer);

      setMessages(prev => [...prev, 'WebRTC: Connection established']);
      setIsConnected(true);

    } catch (error) {
      console.error('Failed to start:', error);
      setMessages(prev => [...prev, `Failed to start: ${error}`]);
    }
  }, [createPeerConnection]);

  const stop = useCallback(async (): Promise<void> => {
    console.log('Stopping recording...');
    setMessages(prev => [...prev, 'Stopping recording...']);

    isRecordingRef.current = false;
    setIsRecording(false);
    setIsConnected(false);

    // Close WebSocket
    if (socket.current) {
      socket.current.close();
      socket.current = null;
    }

    // Close WebRTC connection
    if (pc.current) {
      pc.current.close();
      pc.current = null;
    }

    // Stop microphone stream
    if (localStream.current) {
      localStream.current.getTracks().forEach(track => track.stop());
      localStream.current = null;
    }

    // Cleanup AudioWorklet
    if (ttsWorkletNode.current) {
      ttsWorkletNode.current.disconnect();
      ttsWorkletNode.current = null;
    }

    // Close AudioContext
    if (audioContext.current) {
      await audioContext.current.close();
      audioContext.current = null;
    }

    setMessages(prev => [...prev, 'Recording stopped']);
  }, []);

  const toggleRecording = useCallback(async (): Promise<void> => {
    if (isRecordingRef.current) {
      await stop();
    } else {
      isRecordingRef.current = true;
      setIsRecording(true);
      await start();
    }
  }, [start, stop]);

  // Cleanup function
  const cleanup = useCallback(async () => {
    // Stop recording if active
    if (isRecordingRef.current) {
      await stop();
    }
    
    // Close AudioContext
    if (audioContext.current) {
      await audioContext.current.close();
      audioContext.current = null;
    }
  }, [stop]); // Use ref to avoid stale closures

  return {
    isRecording,
    isConnected,
    isTTSPlaying,
    messages,
    transcription,
    startRecording: start,
    stopRecording: stop,
    toggleRecording,
    cleanup,
    getConnectionState: () => pc.current?.connectionState || 'new',
    getIceConnectionState: () => pc.current?.iceConnectionState || 'new'
  };
}