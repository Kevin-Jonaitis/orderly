import { useState } from 'react';
import { Button, Alert, Badge } from 'react-bootstrap';
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
    <div className="text-center">
      <div className="mb-4">
        <Button
          onClick={toggleRecording}
          disabled={!isConnected && !isRecording}
          variant={isRecording ? 'success' : 'danger'}
          size="lg"
          style={{
            width: '120px',
            height: '120px',
            borderRadius: '50%',
            fontSize: '32px'
          }}
        >
          🎤
        </Button>
        
        <div className="mt-3">
          <div className="mb-2">
            <Badge bg={isConnected ? 'success' : 'danger'}>
              WebSocket: {isConnected ? 'Connected' : 'Disconnected'}
            </Badge>
          </div>
          <div>
            <Badge bg={isRecording ? 'primary' : 'secondary'}>
              Status: {isRecording ? 'Recording...' : 'Ready'}
            </Badge>
          </div>
        </div>
      </div>

      {transcription && (
        <Alert variant="info" className="my-3">
          <Alert.Heading>Current Transcription:</Alert.Heading>
          <p className="mb-0">"{transcription}"</p>
        </Alert>
      )}

      <div className="text-start">
        <h5>Activity Log:</h5>
        <div className="bg-light p-3 rounded" style={{ maxHeight: '200px', overflowY: 'auto' }}>
          {messages.slice(-5).map((message, index) => (
            <div key={index} className="py-1 border-bottom">
              <small>{message}</small>
            </div>
          ))}
          {messages.length === 0 && (
            <small className="text-muted">No activity yet...</small>
          )}
        </div>
      </div>
    </div>
  );
}