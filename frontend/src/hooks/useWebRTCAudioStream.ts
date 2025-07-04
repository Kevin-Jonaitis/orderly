import { useRef, useState, useCallback } from 'react';
import { AudioMessage } from '../types/order';

// Direct TypeScript port of aiortc client.js
export function useWebRTCAudioStream() {
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<string[]>([]);
  const [transcription, setTranscription] = useState<string>('');
  
  // WebRTC refs - matching aiortc client.js variable names
  const pc = useRef<RTCPeerConnection | null>(null);
  const localStream = useRef<MediaStream | null>(null);

  const handleAudioMessage = (message: AudioMessage) => {
    console.log('Received audio message:', message);
    
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

  /**
   * Create peer connection - exact copy of aiortc client.js createPeerConnection()
   */
  const createPeerConnection = useCallback((): RTCPeerConnection => {
    const config: RTCConfiguration = {
      iceServers: []
    };

    const peerConnection = new RTCPeerConnection(config);

    // Connection state change handler
    peerConnection.addEventListener('connectionstatechange', () => {
      console.log('Connection state:', peerConnection.connectionState);
      
      if (peerConnection.connectionState === 'connected') {
        setIsConnected(true);
        setMessages(prev => [...prev, 'WebRTC: Connected']);
      } else if (peerConnection.connectionState === 'failed' || peerConnection.connectionState === 'disconnected') {
        setIsConnected(false);
        setMessages(prev => [...prev, 'WebRTC: Disconnected']);
      }
    });

    // ICE connection state change handler
    peerConnection.addEventListener('iceconnectionstatechange', () => {
      console.log('ICE connection state:', peerConnection.iceConnectionState);
    });

    // ICE gathering state change handler
    peerConnection.addEventListener('icegatheringstatechange', () => {
      console.log('ICE gathering state:', peerConnection.iceGatheringState);
    });

    // Signaling state change handler
    peerConnection.addEventListener('signalingstatechange', () => {
      console.log('Signaling state:', peerConnection.signalingState);
    });

    return peerConnection;
  }, []);

  /**
   * Negotiate - exact copy of aiortc client.js negotiate() function
   */
  const negotiate = useCallback((): Promise<void> => {
    return pc.current!.createOffer().then((offer) => {
      return pc.current!.setLocalDescription(offer);
    }).then(() => {
      // Wait for ICE gathering to complete - exact aiortc pattern
      return new Promise<void>((resolve) => {
        if (pc.current!.iceGatheringState === 'complete') {
          resolve();
        } else {
          function checkState() {
            if (pc.current!.iceGatheringState === 'complete') {
              pc.current!.removeEventListener('icegatheringstatechange', checkState);
              resolve();
            }
          }
          pc.current!.addEventListener('icegatheringstatechange', checkState);
        }
      });
    }).then(() => {
      const offer = pc.current!.localDescription!;
      
      return fetch('http://localhost:8002/api/webrtc/offer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
        }),
      });
    }).then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    }).then((answer) => {
      return pc.current!.setRemoteDescription(answer);
    });
  }, []);

  /**
   * Start - exact copy of aiortc client.js start() function pattern
   */
  const start = useCallback(async (): Promise<void> => {
    console.log('Starting WebRTC connection...');
    setMessages(prev => [...prev, 'Starting WebRTC connection...']);

    // Create peer connection
    pc.current = createPeerConnection();

    // Get user media - audio only for our use case
    const constraints: MediaStreamConstraints = {
      audio: {
        sampleRate: 16000, // Match backend and NeMo STT sample rate
        channelCount: 1, // Mono audio
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      },
      video: false
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      localStream.current = stream;
      
      // Debug: log actual audio track settings
      const audioTrack = stream.getAudioTracks()[0];
      if (audioTrack) {
        console.log('Actual audio track settings:', audioTrack.getSettings());
      }

      console.log('Microphone access granted');
      setMessages(prev => [...prev, 'Microphone access granted']);

      // Add tracks to peer connection
      stream.getTracks().forEach((track) => {
        pc.current!.addTrack(track, stream);
      });

      // Start negotiation
      await negotiate();

      setIsRecording(true);
      console.log('WebRTC recording started');
      setMessages(prev => [...prev, 'WebRTC: Recording started']);

    } catch (error) {
      console.error('Error starting WebRTC:', error);
      setMessages(prev => [...prev, `Error: ${(error as any).message}`]);
      await stop();
      throw error;
    }
  }, [createPeerConnection, negotiate]);

  /**
   * Stop - exact copy of aiortc client.js stop() function
   */
  const stop = useCallback(async (): Promise<void> => {
    console.log('Stopping WebRTC connection...');
    
    setIsRecording(false);
    setIsConnected(false);

    // Stop local stream tracks
    if (localStream.current) {
      localStream.current.getTracks().forEach((track) => {
        track.stop();
      });
      localStream.current = null;
    }

    // Close peer connection
    if (pc.current) {
      pc.current.close();
      pc.current = null;
    }

    console.log('WebRTC connection stopped');
    setMessages(prev => [...prev, 'WebRTC: Connection stopped']);
  }, []);

  const toggleRecording = async () => {
    try {
      if (isRecording) {
        await stop();
      } else {
        await start();
      }
    } catch (error) {
      console.error('Error toggling recording:', error);
      alert(`Could not ${isRecording ? 'stop' : 'start'} recording: ${(error as any).message}`);
    }
  };

  // Cleanup function
  const cleanup = useCallback(async () => {
    if (isRecording) {
      await stop();
    }
  }, [isRecording, stop]);

  return {
    isRecording,
    isConnected,
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