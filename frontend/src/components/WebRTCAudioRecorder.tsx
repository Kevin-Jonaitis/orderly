import React, { useEffect } from 'react';
import { Button, Card, ListGroup, Badge } from 'react-bootstrap';
import { useWebRTCAudioStream } from '../hooks/useWebRTCAudioStream';

export function WebRTCAudioRecorder() {
  const {
    isRecording,
    isConnected,
    messages,
    transcription,
    toggleRecording,
    cleanup,
    getConnectionState,
    getIceConnectionState
  } = useWebRTCAudioStream();

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  const getStatusBadge = () => {
    if (isRecording && isConnected) {
      return <Badge bg="success">Connected & Recording</Badge>;
    } else if (isRecording) {
      return <Badge bg="warning">Connecting...</Badge>;
    } else {
      return <Badge bg="secondary">Stopped</Badge>;
    }
  };

  return (
    <Card>
      <Card.Header>
        <div className="d-flex justify-content-between align-items-center">
          <h5 className="mb-0">🎤 WebRTC Audio Recorder</h5>
          {getStatusBadge()}
        </div>
      </Card.Header>
      
      <Card.Body>
        <div className="mb-3">
          <Button
            variant={isRecording ? "danger" : "primary"}
            onClick={toggleRecording}
            size="lg"
            className="w-100"
          >
            {isRecording ? "🛑 Stop Recording" : "🎤 Start Recording"}
          </Button>
        </div>

        {/* Connection Status */}
        <div className="mb-3">
          <small className="text-muted">
            Connection: {getConnectionState()} | ICE: {getIceConnectionState()}
          </small>
        </div>

        {/* Current Transcription */}
        {transcription && (
          <div className="mb-3">
            <strong>Current transcription:</strong>
            <div className="p-2 bg-light rounded">
              "{transcription}"
            </div>
          </div>
        )}

        {/* Messages Log */}
        <div>
          <h6>Activity Log:</h6>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            <ListGroup variant="flush">
              {messages.slice(-10).map((message, index) => (
                <ListGroup.Item key={index} className="py-1 px-0">
                  <small>{message}</small>
                </ListGroup.Item>
              ))}
              {messages.length === 0 && (
                <ListGroup.Item className="py-1 px-0">
                  <small className="text-muted">No activity yet...</small>
                </ListGroup.Item>
              )}
            </ListGroup>
          </div>
        </div>

        {/* Debug Info */}
        <div className="mt-3 pt-3 border-top">
          <small className="text-muted">
            <strong>WebRTC Audio Streaming:</strong> Real-time audio to AI backend
            <br />
            • Browser microphone → WebRTC → Speech-to-Text → LLM → Text-to-Speech
          </small>
        </div>
      </Card.Body>
    </Card>
  );
}