import React, { useState } from 'react';
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
    <div className="audio-recorder">
      <div className="recorder-controls">
        <button
          onClick={toggleRecording}
          className={`record-button ${isRecording ? 'recording' : ''}`}
          disabled={!isConnected && !isRecording}
        >
          <div className="microphone-icon">
            🎤
          </div>
          <span className="record-text">
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </span>
        </button>
        
        <div className="status-indicators">
          <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            WebSocket: {isConnected ? 'Connected' : 'Disconnected'}
          </div>
          <div className={`status-indicator ${isRecording ? 'recording' : 'idle'}`}>
            Status: {isRecording ? 'Recording...' : 'Ready'}
          </div>
        </div>
      </div>

      {transcription && (
        <div className="transcription">
          <h3>Current Transcription:</h3>
          <p>"{transcription}"</p>
        </div>
      )}

      <div className="message-log">
        <h3>Activity Log:</h3>
        <div className="messages">
          {messages.slice(-5).map((message, index) => (
            <div key={index} className="message">
              {message}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}