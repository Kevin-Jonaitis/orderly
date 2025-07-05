import { useRef, useState, useCallback } from 'react';
import { AudioMessage } from '../types/order';

// Clean rewrite based on aiortc client.js
export function useWebRTCAudioStream() {
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<string[]>([]);
  const [transcription, setTranscription] = useState<string>('');
  const [isTTSPlaying, setIsTTSPlaying] = useState(false);
  
  // WebRTC refs
  const pc = useRef<RTCPeerConnection | null>(null);
  const localStream = useRef<MediaStream | null>(null);
  const ttsAudioElement = useRef<HTMLAudioElement | null>(null);

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
   * Create peer connection - based on aiortc client.js
   */
  const createPeerConnection = useCallback((): RTCPeerConnection => {
    const config: RTCConfiguration = {
      iceServers: [],
      rtcpMuxPolicy: 'require',
      bundlePolicy: 'max-bundle'
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

    // Handle incoming tracks (audio from backend)
    peerConnection.ontrack = (event) => {
      console.log('Received track:', event.track.kind, 'from transceiver:', event.transceiver?.direction);
      
      // Log the full event object
      console.log('🎵 Full track event:', event);
      console.log('🎵 Event transceiver:', event.transceiver);
      console.log('🎵 Event track:', event.track);
      
      if (event.track.kind === 'audio') {
        // Check if this is our own microphone track being echoed back
        const isOurTrack = event.track.id === localStream.current?.getAudioTracks()[0]?.id;
        const trackDirection = event.transceiver?.direction;
        
        console.log('🎵 Track details:', {
          trackId: event.track.id,
          ourTrackId: localStream.current?.getAudioTracks()[0]?.id,
          isOurTrack: isOurTrack,
          transceiverDirection: trackDirection,
          trackState: event.track.readyState,
          transceiverIndex: pc.current?.getTransceivers().findIndex(t => t.receiver.track === event.track) ?? -1
        });
        
        // Skip if this appears to be our own microphone track
        if (isOurTrack) {
          console.warn('🎤 Ignoring our own microphone track to prevent echo');
          setMessages(prev => [...prev, '🎤 Ignored own microphone track']);
          return;
        }
        
        // This should be audio from the backend (TTS or other audio)
        const mid = event.transceiver?.mid;
        
        if (mid === "0") {
          console.log('🎤 Ignoring audio from MID 0 (processor track)');
          setMessages(prev => [...prev, '🎤 Processor track received (MID 0) - not playing']);
          return; // Don't play audio from MID 0
        } else if (mid === "1") {
          console.log('🎵 Playing audio from MID 1 (TTS response track)');
          console.log('🎵 Track details:', {
            id: event.track.id,
            kind: event.track.kind,
            readyState: event.track.readyState,
            enabled: event.track.enabled
          });
          setMessages(prev => [...prev, '🎵 TTS response track received (MID 1)']);
        } else {
          console.log(`🎵 Ignoring audio from MID ${mid}`);
          setMessages(prev => [...prev, `🎵 Audio from MID ${mid} - not playing`]);
          return; // Don't play audio from unknown MID
        }
        
        // Create audio element for backend audio (only for MID 1)
        const audio = new Audio();
        audio.srcObject = new MediaStream([event.track]);
        audio.autoplay = true;
        audio.volume = 0.8;
        
        // Debug: Log audio element details
        console.log('🎵 Audio element created:', {
          srcObject: audio.srcObject,
          autoplay: audio.autoplay,
          volume: audio.volume,
          readyState: audio.readyState,
          paused: audio.paused,
          currentTime: audio.currentTime,
          duration: audio.duration
        });
        
        // Debug: Log WebRTC stats to see if audio is flowing
        setTimeout(() => {
          pc.current?.getStats().then(stats => {
            console.log('🎵 WebRTC Stats:');
            stats.forEach(report => {
              if (report.type === 'inbound-rtp' && report.kind === 'audio') {
                console.log('🎵 Audio RTP Stats:', {
                  packetsReceived: report.packetsReceived,
                  bytesReceived: report.bytesReceived,
                  packetsLost: report.packetsLost,
                  jitter: report.jitter,
                  timestamp: report.timestamp
                });
              }
            });
          });
        }, 2000);
        
        // Store reference to prevent garbage collection
        ttsAudioElement.current = audio;
        
        // Add event listeners to debug audio playback
        audio.onloadstart = () => {
          console.log('🎵 Audio: loadstart event');
          setMessages(prev => [...prev, '🎵 Audio: loadstart']);
        };
        
        audio.oncanplay = () => {
          console.log('🎵 Audio: canplay event');
          setMessages(prev => [...prev, '🎵 Audio: canplay']);
        };
        
        audio.onplay = () => {
          console.log('🎵 Audio: play event');
          setIsTTSPlaying(true);
          setMessages(prev => [...prev, '🎵 Audio: playing']);
        };
        
        audio.onpause = () => {
          console.log('🎵 Audio: pause event');
          setIsTTSPlaying(false);
          setMessages(prev => [...prev, '🎵 Audio: paused']);
        };
        
        audio.onended = () => {
          console.log('🎵 Audio: ended event');
          setIsTTSPlaying(false);
          setMessages(prev => [...prev, '🎵 Audio: ended']);
        };
        
        audio.onerror = (e) => {
          console.error('🎵 Audio: error event', e);
          setMessages(prev => [...prev, '🎵 Audio: error']);
        };
        
        // Add track event listeners to monitor data flow
        event.track.onended = () => {
          console.log('🎵 Track ended');
          setMessages(prev => [...prev, '🎵 Track ended']);
        };
        
        event.track.onmute = () => {
          console.log('🎵 Track muted');
          setMessages(prev => [...prev, '🎵 Track muted']);
        };
        
        event.track.onunmute = () => {
          console.log('🎵 Track unmuted');
          setMessages(prev => [...prev, '🎵 Track unmuted']);
        };
        
        audio.onended = () => {
          setIsTTSPlaying(false);
          setMessages(prev => [...prev, '🎵 Backend Audio: Ended']);
        };
        
        audio.onerror = (error) => {
          console.error('Backend audio error:', error);
          setMessages(prev => [...prev, '❌ Backend Audio: Error']);
        };
        
        console.log('🎵 Backend audio element created and configured');
        setMessages(prev => [...prev, '🎵 Backend Audio: Ready to play']);
      }
    };

    return peerConnection;
  }, []);

  /**
   * Negotiate - based on aiortc client.js
   */
  const negotiate = useCallback(async (): Promise<void> => {
    if (!pc.current) return;

    console.log('Starting negotiation...');
    setMessages(prev => [...prev, 'Starting negotiation...']);

    try {
      await pc.current.setLocalDescription(await pc.current.createOffer());
      
      // Wait for ICE gathering to complete
      await new Promise<void>((resolve) => {
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

      const offer = pc.current!.localDescription!;
      
              console.log('📤 SDP Offer created');
        setMessages(prev => [...prev, '📤 SDP Offer created']);
      
      const response = await fetch('http://localhost:8002/api/webrtc/offer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

              const answer = await response.json();
        console.log('📥 SDP Answer received');
        setMessages(prev => [...prev, '📥 SDP Answer received']);
      
      await pc.current!.setRemoteDescription(answer);
    } catch (error) {
      console.error('Negotiation error:', error);
      setMessages(prev => [...prev, `Negotiation error: ${(error as any).message}`]);
      throw error;
    }
  }, []);

  /**
   * Start - based on aiortc client.js
   */
  const start = useCallback(async (): Promise<void> => {
    console.log('Starting WebRTC connection...');
    setMessages(prev => [...prev, 'Starting WebRTC connection...']);

    // Create peer connection
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
      
      console.log('Microphone access granted');
      setMessages(prev => [...prev, 'Microphone access granted']);

      // Create two transceivers: transceiver 0 for microphone, transceiver 1 for TTS audio
      
      // Transceiver 0: for sending microphone audio to backend
      const micTransceiver = pc.current.addTransceiver('audio', {
        direction: 'sendrecv'
      });
      console.log('🎤 Added microphone transceiver (index 0, MID:', micTransceiver.mid, ')');
      setMessages(prev => [...prev, `🎤 Microphone transceiver added (MID: ${micTransceiver.mid})`]);

      // Transceiver 1: for receiving TTS audio from backend
      const ttsTransceiver = pc.current.addTransceiver('audio', {
        direction: 'sendrecv'
      });
      console.log('🎵 Added TTS transceiver (index 1, MID:', ttsTransceiver.mid, ')');
      setMessages(prev => [...prev, `🎵 TTS transceiver added (MID: ${ttsTransceiver.mid})`]);

      // Add the microphone track to transceiver 0
      const audioTrack = stream.getAudioTracks()[0];
      if (audioTrack) {
        micTransceiver.sender.replaceTrack(audioTrack);
        console.log('🎤 Added microphone track to transceiver 0');
        setMessages(prev => [...prev, '🎤 Microphone track added to transceiver 0']);
      }

              // Log transceiver setup
        console.log('🔍 Transceivers setup complete');
        setMessages(prev => [...prev, '🔍 Transceivers setup complete']);

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
   * Stop - based on aiortc client.js
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