import React, { useEffect } from 'react';
import { Button } from 'react-bootstrap';
import { useWebRTCAudioStream } from '../hooks/useWebRTCAudioStream';

export function WebRTCAudioRecorder() {
  const {
    isRecording,
    toggleRecording,
    cleanup
  } = useWebRTCAudioStream();

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  return (
    <div className="text-center">
      <Button
        variant={isRecording ? "danger" : "primary"}
        onClick={toggleRecording}
        size="lg"
        className="px-5 py-3"
      >
        {isRecording ? "Speak Your Order" : "Click to Start Order"}
      </Button>
    </div>
  );
}