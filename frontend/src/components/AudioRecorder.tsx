import { useState } from 'react';
import { Button, Alert, Badge, Card, Stack, Container } from 'react-bootstrap';
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
    <Container className="text-center">
      <Stack gap={4}>
        <Stack gap={3} className="align-items-center">
          <Button
            onClick={toggleRecording}
            disabled={!isConnected && !isRecording}
            variant={isRecording ? 'success' : 'danger'}
            size="lg"
            className="rounded-circle p-4 fs-1 d-flex align-items-center justify-content-center"
            style={{ width: '120px', height: '120px' }}
          >
            🎤
          </Button>
          
          <Stack gap={2} className="align-items-center">
            <Badge bg={isConnected ? 'success' : 'danger'} className="w-75 p-2">
              WebSocket: {isConnected ? 'Connected' : 'Disconnected'}
            </Badge>
            <Badge bg={isRecording ? 'primary' : 'secondary'} className="w-75 p-2">
              Status: {isRecording ? 'Recording...' : 'Ready'}
            </Badge>
          </Stack>
        </Stack>

        {transcription && (
          <Alert variant="info">
            <Alert.Heading>Current Transcription:</Alert.Heading>
            <Alert.Link>{transcription}</Alert.Link>
          </Alert>
        )}

        <Container className="text-start">
          <Card.Title as="h5">Activity Log:</Card.Title>
          <Card className="bg-light">
            <Card.Body className="overflow-auto">
              <Stack gap={1}>
                {messages.slice(-5).map((message, index) => (
                  <Card.Text key={index} as="small" className="border-bottom py-1">
                    {message}
                  </Card.Text>
                ))}
                {messages.length === 0 && (
                  <Card.Text as="small" className="text-muted">
                    No activity yet...
                  </Card.Text>
                )}
              </Stack>
            </Card.Body>
          </Card>
        </Container>
      </Stack>
    </Container>
  );
}