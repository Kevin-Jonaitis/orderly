import { useRef, useState, useCallback } from 'react';
import { AudioMessage } from '../types/order';

// Direct TypeScript port of aiortc client.js
export function useWebRTCAudioStream() {
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<string[]>([]);
  const [transcription, setTranscription] = useState<string>('');
  const [isTTSPlaying, setIsTTSPlaying] = useState(false);
  
  // WebRTC refs - matching aiortc client.js variable names
  const pc = useRef<RTCPeerConnection | null>(null);
  const localStream = useRef<MediaStream | null>(null);
  const ttsAudioElement = useRef<HTMLAudioElement | null>(null);
  
  // Frame debugging refs
  const frameStats = useRef({
    sentFrames: 0,
    receivedFrames: 0,
    droppedFrames: 0,
    lastFrameTime: 0,
    frameIntervals: [] as number[],
    bufferUnderruns: 0,
    bufferOverruns: 0
  });

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
      iceServers: [],
      // Enable better audio codecs for higher quality
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
      console.log('Received track:', event.track.kind);
      
      if (event.track.kind === 'audio') {
        // Check if this is our own microphone track being echoed back
        const isOurTrack = event.track.id === localStream.current?.getAudioTracks()[0]?.id;
		console.log("OUR LOCAL STREAM", localStream.current?.getAudioTracks());
		console.log("EVENT", event)
        const trackDirection = event.receiver?.track?.readyState;
        
        console.log('🎵 Track details:', {
          trackId: event.track.id,
          ourTrackId: localStream.current?.getAudioTracks()[0]?.id,
          isOurTrack: isOurTrack,
          trackDirection: trackDirection,
          trackState: event.track.readyState
        });
        
        // Skip if this appears to be our own microphone track
        if (isOurTrack) {
          console.warn('🎤 Ignoring our own microphone track to prevent echo');
          setMessages(prev => [...prev, '🎤 Ignored own microphone track']);
          return;
        }
		        
        // This should be audio from the backend (TTS or other audio)
        console.log('🎵 Playing audio track from backend');
        
        // Create audio element for backend audio
        const audio = new Audio();
        audio.srcObject = new MediaStream([event.track]);
        audio.autoplay = true;
        audio.volume = 0.8;
        
        // Store reference to prevent garbage collection
        ttsAudioElement.current = audio;
        
        // Track audio playing state
        audio.onplay = () => {
          setIsTTSPlaying(true);
          setMessages(prev => [...prev, '🎵 Backend Audio: Playing']);
        };
        
        audio.onpause = () => {
          setIsTTSPlaying(false);
          setMessages(prev => [...prev, '🎵 Backend Audio: Paused']);
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

    // Monitor connection quality for frame dropping issues
    const monitorConnectionQuality = async () => {
      try {
        const stats = await peerConnection.getStats();
        let audioStats: RTCStatsReport | null = null;
        
        stats.forEach((report) => {
          if (report.type === 'outbound-rtp' && (report as any).mediaType === 'audio') {
            audioStats = report;
          }
        });
        
        if (audioStats) {
          const packetsLost = (audioStats as any).packetsLost || 0;
          const packetsSent = (audioStats as any).packetsSent || 0;
          const jitter = (audioStats as any).jitter || 0;
          const bytesSent = (audioStats as any).bytesSent || 0;
          
          // Calculate packet loss percentage
          const lossPercentage = packetsSent > 0 ? (packetsLost / packetsSent) * 100 : 0;
          
          if (lossPercentage > 1) {
            console.warn(`🎤 High packet loss: ${lossPercentage.toFixed(2)}% (${packetsLost}/${packetsSent})`);
            setMessages(prev => [...prev, `⚠️ Packet loss: ${lossPercentage.toFixed(1)}%`]);
          }
          
          if (jitter > 0.02) { // 20ms jitter threshold
            console.warn(`🎤 High jitter: ${(jitter * 1000).toFixed(1)}ms`);
            setMessages(prev => [...prev, `⚠️ High jitter: ${(jitter * 1000).toFixed(1)}ms`]);
          }
          
          // Log stats every 5 seconds
          if (frameStats.current.sentFrames % 50 === 0) {
            console.log(`🎤 Connection stats: Loss=${lossPercentage.toFixed(2)}%, Jitter=${(jitter * 1000).toFixed(1)}ms, Bytes=${(bytesSent / 1024).toFixed(1)}KB`);
          }
        }
      } catch (error) {
        console.warn('Could not get connection stats:', error);
      }
    };
    
    // Monitor connection quality every 2 seconds
    const qualityInterval = setInterval(monitorConnectionQuality, 2000);
    
    // Clean up interval on connection close
    peerConnection.addEventListener('connectionstatechange', () => {
      if (peerConnection.connectionState === 'closed' || peerConnection.connectionState === 'failed') {
        clearInterval(qualityInterval);
      }
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
      
      // Debug: log actual audio track settings
      const audioTrack = stream.getAudioTracks()[0];
      if (audioTrack) {
        console.log('Actual audio track settings:', audioTrack.getSettings());
        
        // Monitor frame dropping and track state
        audioTrack.onended = () => {
          console.warn('🎤 Audio track ended unexpectedly');
          setMessages(prev => [...prev, '❌ Audio track ended']);
        };
        
        audioTrack.onmute = () => {
          console.warn('🎤 Audio track muted');
          setMessages(prev => [...prev, '🔇 Audio track muted']);
        };
        
        audioTrack.onunmute = () => {
          console.log('🎤 Audio track unmuted');
          setMessages(prev => [...prev, '🔊 Audio track unmuted']);
        };
        
        // Monitor track enabled state
        setInterval(() => {
          if (!audioTrack.enabled) {
            console.warn('🎤 Audio track disabled - this could cause frame dropping');
            setMessages(prev => [...prev, '⚠️ Audio track disabled']);
          }
        }, 1000);
      }

      console.log('Microphone access granted');
      setMessages(prev => [...prev, 'Microphone access granted']);

      // Add tracks to peer connection
      stream.getTracks().forEach((track) => {
        const sender = pc.current!.addTrack(track, stream);
        
        // Optimize audio encoding parameters for high quality
        if (track.kind === 'audio') {
          // Set encoding parameters after a short delay to ensure track is ready
          setTimeout(async () => {
            try {
              const params = sender.getParameters();
              if (params.encodings && params.encodings.length > 0) {
                params.encodings[0].maxBitrate = 512_000; // 512 kbps — high for Opus
                params.encodings[0].priority = "high";     // or "very-high" (not always honored)
                params.encodings[0].networkPriority = "high";  // hint to the transport layer
                
                await sender.setParameters(params);
                console.log('🎤 Audio encoding parameters optimized:', {
                  maxBitrate: params.encodings[0].maxBitrate,
                  priority: params.encodings[0].priority,
                  networkPriority: params.encodings[0].networkPriority
                });
                setMessages(prev => [...prev, '🎤 Audio quality optimized']);
                
                // Monitor frame processing
                const monitorFrames = () => {
                  frameStats.current.sentFrames++;
                  const now = Date.now();
                  
                  if (frameStats.current.lastFrameTime > 0) {
                    const interval = now - frameStats.current.lastFrameTime;
                    frameStats.current.frameIntervals.push(interval);
                    
                    // Keep only last 100 intervals
                    if (frameStats.current.frameIntervals.length > 100) {
                      frameStats.current.frameIntervals.shift();
                    }
                    
                    // Detect frame drops (intervals > 50ms are suspicious)
                    if (interval > 50) {
                      frameStats.current.droppedFrames++;
                      console.warn(`🎤 Potential frame drop detected: ${interval}ms interval`);
                      setMessages(prev => [...prev, `⚠️ Frame drop: ${interval}ms`]);
                    }
                  }
                  
                  frameStats.current.lastFrameTime = now;
                  
                  // Log frame stats every 100 frames
                  if (frameStats.current.sentFrames % 100 === 0) {
                    const avgInterval = frameStats.current.frameIntervals.reduce((a, b) => a + b, 0) / frameStats.current.frameIntervals.length;
                    console.log(`🎤 Frame stats: Sent=${frameStats.current.sentFrames}, Dropped=${frameStats.current.droppedFrames}, AvgInterval=${avgInterval.toFixed(1)}ms`);
                  }
                };
                
                // Monitor frames every 20ms (typical audio frame rate)
                const frameInterval = setInterval(monitorFrames, 20);
                
                // Clean up on connection close
                pc.current!.addEventListener('connectionstatechange', () => {
                  if (pc.current!.connectionState === 'closed' || pc.current!.connectionState === 'failed') {
                    clearInterval(frameInterval);
                  }
                });
              }
            } catch (error) {
              console.warn('Could not set audio encoding parameters:', error);
            }
          }, 100);
        }
      });

      // Set preferred codecs for better audio quality after adding tracks
      pc.current!.getTransceivers().forEach(transceiver => {
        if (transceiver.receiver.track?.kind === 'audio') {
          const capabilities = RTCRtpReceiver.getCapabilities('audio');
          if (capabilities) {
            // Prefer Opus with 48kHz for highest quality
            const opusCodec = capabilities.codecs.find(codec => 
              codec.mimeType === 'audio/opus' && codec.clockRate === 48000
            );
            if (opusCodec) {
              transceiver.setCodecPreferences([opusCodec]);
              console.log('🎤 Set preferred audio codec: Opus 48kHz');
              setMessages(prev => [...prev, '🎤 Audio codec optimized: Opus 48kHz']);
            }
          }
        }
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