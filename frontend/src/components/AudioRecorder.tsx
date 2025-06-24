import { useState } from 'react';
import { useAudioStream } from '../hooks/useAudioStream';
import { AudioMessage } from '../types/order';

export function AudioRecorder() {
  const [messages, setMessages] = useState<string[]>([]);
  const [transcription, setTranscription] = useState<string>('');

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

  const { isRecording, isConnected, toggleRecording } = useAudioStream(handleAudioMessage);

  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ marginBottom: '20px' }}>
        <button
          onClick={toggleRecording}
          disabled={!isConnected && !isRecording}
          style={{
            width: '120px',
            height: '120px',
            borderRadius: '50%',
            border: 'none',
            backgroundColor: isRecording ? '#28a745' : '#dc3545',
            color: 'white',
            fontSize: '24px',
            cursor: !isConnected && !isRecording ? 'not-allowed' : 'pointer',
            opacity: !isConnected && !isRecording ? 0.5 : 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px'
          }}
        >
          <div style={{ fontSize: '32px' }}>🎤</div>
          <div style={{ fontSize: '12px' }}>
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </div>
        </button>
        
        <div style={{ marginTop: '20px' }}>
          <div style={{ 
            padding: '5px 10px', 
            margin: '5px 0',
            borderRadius: '4px',
            backgroundColor: isConnected ? '#d4edda' : '#f8d7da',
            color: isConnected ? '#155724' : '#721c24',
            fontSize: '14px'
          }}>
            WebSocket: {isConnected ? 'Connected' : 'Disconnected'}
          </div>
          <div style={{ 
            padding: '5px 10px', 
            margin: '5px 0',
            borderRadius: '4px',
            backgroundColor: isRecording ? '#cce5ff' : '#f8f9fa',
            color: isRecording ? '#004085' : '#495057',
            fontSize: '14px'
          }}>
            Status: {isRecording ? 'Recording...' : 'Ready'}
          </div>
        </div>
      </div>

      {transcription && (
        <div style={{ 
          backgroundColor: '#e7f3ff', 
          border: '1px solid #0ea5e9', 
          borderRadius: '8px', 
          padding: '16px', 
          margin: '16px 0' 
        }}>
          <h3 style={{ margin: '0 0 8px 0', color: '#0c4a6e' }}>Current Transcription:</h3>
          <p style={{ margin: 0 }}>"{transcription}"</p>
        </div>
      )}

      <div style={{ textAlign: 'left', marginTop: '20px' }}>
        <h3 style={{ marginBottom: '16px', color: '#374151' }}>Activity Log:</h3>
        <div style={{ 
          backgroundColor: '#f9fafb', 
          borderRadius: '8px', 
          padding: '16px', 
          maxHeight: '200px', 
          overflowY: 'auto',
          border: '1px solid #e5e7eb'
        }}>
          {messages.slice(-5).map((message, index) => (
            <div key={index} style={{ 
              padding: '8px 0', 
              borderBottom: index < messages.slice(-5).length - 1 ? '1px solid #e5e7eb' : 'none',
              fontSize: '14px'
            }}>
              {message}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}