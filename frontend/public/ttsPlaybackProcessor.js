class TTSPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferQueue = [];
    this.readOffset = 0;
    this.samplesRemaining = 0;
    this.isPlaying = false;
    this.firstDataReceived = false;
    this.firstDataTime = null;

    // Listen for incoming messages
    this.port.onmessage = (event) => {
      // Check if this is a control message (object with a "type" property).
      if (event.data && typeof event.data === "object" && event.data.type === "clear") {
        // Clear the TTS buffer and reset playback state.
        this.bufferQueue = [];
        this.readOffset = 0;
        this.samplesRemaining = 0;
        this.isPlaying = false;
        console.log('🎵 [AudioWorklet] Buffer cleared');
        return;
      }
      
      // Handle audio data with timestamp
      if (event.data && typeof event.data === "object" && event.data.audioData) {
        // Track first data received
        if (!this.firstDataReceived) {
          this.firstDataReceived = true;
          this.firstDataTime = event.data.timestamp;
          console.log(`⏱️ [AudioWorklet] First audio data received at timestamp: ${this.firstDataTime}`);
        }
        
        // Add audio data to buffer
        this.bufferQueue.push(event.data.audioData);
        this.samplesRemaining += event.data.audioData.length;
        return;
      }
      
      // Fallback: assume it's a direct PCM chunk (for backward compatibility)
      if (event.data instanceof Int16Array) {
        // Track first data received
        if (!this.firstDataReceived) {
          this.firstDataReceived = true;
          this.firstDataTime = currentTime;
          console.log(`⏱️ [AudioWorklet] First audio data received at: ${this.firstDataTime.toFixed(6)}s`);
        }
        
        this.bufferQueue.push(event.data);
        this.samplesRemaining += event.data.length;
      }
    };
  }

  process(inputs, outputs) {
    const outputChannel = outputs[0][0];

    if (this.samplesRemaining === 0) {
      outputChannel.fill(0);
      if (this.isPlaying) {
        this.isPlaying = false;
        this.port.postMessage({ type: 'ttsPlaybackStopped' });
      }
      return true;
    }

    if (!this.isPlaying) {
      this.isPlaying = true;
      this.port.postMessage({ type: 'ttsPlaybackStarted' });
    }

    let outIdx = 0;
    while (outIdx < outputChannel.length && this.bufferQueue.length > 0) {
      const currentBuffer = this.bufferQueue[0];
      const sampleValue = currentBuffer[this.readOffset] / 32768;
      outputChannel[outIdx++] = sampleValue;

      this.readOffset++;
      this.samplesRemaining--;

      if (this.readOffset >= currentBuffer.length) {
        this.bufferQueue.shift();
        this.readOffset = 0;
      }
    }

    while (outIdx < outputChannel.length) {
      outputChannel[outIdx++] = 0;
    }

    return true;
  }
}

registerProcessor('tts-playback-processor', TTSPlaybackProcessor); 