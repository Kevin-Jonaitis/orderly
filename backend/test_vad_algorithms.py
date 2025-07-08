#!/usr/bin/env python3
"""
Voice Activity Detection (VAD) Algorithm Comparison
Tests multiple silence detection techniques against test_VAD.wav
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import time
from typing import List, Tuple, Dict
import json

# Try to import optional VAD libraries
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    print("⚠️  webrtcvad not available. Install with: pip install webrtcvad")

try:
    import torch
    import torchaudio
    TORCH_VAD_AVAILABLE = True
except ImportError:
    TORCH_VAD_AVAILABLE = False
    print("⚠️  torch/torchaudio not available. Install with: pip install torch torchaudio")

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available. Install with: pip install scipy")

class VADTester:
    """Test multiple VAD algorithms on audio file"""
    
    def __init__(self, audio_file: str):
        self.audio_file = audio_file
        self.audio_data = None
        self.sample_rate = None
        self.results = {}
        
        # Load audio
        self._load_audio()
        
        # VAD parameters
        self.frame_duration_ms = 20  # 20ms frames
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        # Create directories
        self.cut_dir = Path("cut_results")
        self.cut_dir.mkdir(exist_ok=True)
        
    def _load_audio(self):
        """Load audio file"""
        print(f"📁 Loading audio file: {self.audio_file}")
        self.audio_data, self.sample_rate = sf.read(self.audio_file)
        
        # Convert to mono if stereo
        if len(self.audio_data.shape) > 1:
            self.audio_data = np.mean(self.audio_data, axis=1)
        
        print(f"🎵 Audio: {len(self.audio_data)} samples, {self.sample_rate}Hz, {len(self.audio_data)/self.sample_rate:.2f}s")
    
    def rms_vad(self, threshold: float = 0.01) -> List[bool]:
        """RMS-based VAD (current implementation)"""
        print("🔍 Testing RMS-based VAD...")
        
        speech_frames = []
        for i in range(0, len(self.audio_data), self.frame_size):
            frame = self.audio_data[i:i+self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            
            rms = np.sqrt(np.mean(frame**2))
            speech_frames.append(rms > threshold)
        
        return speech_frames
    
    def energy_vad(self, threshold: float = 0.005) -> List[bool]:
        """Energy-based VAD"""
        print("🔍 Testing Energy-based VAD...")
        
        speech_frames = []
        for i in range(0, len(self.audio_data), self.frame_size):
            frame = self.audio_data[i:i+self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            
            energy = np.sum(frame**2)
            speech_frames.append(energy > threshold)
        
        return speech_frames
    
    def zero_crossing_vad(self, threshold: float = 0.1) -> List[bool]:
        """Zero-crossing rate VAD"""
        print("🔍 Testing Zero-crossing VAD...")
        
        speech_frames = []
        for i in range(0, len(self.audio_data), self.frame_size):
            frame = self.audio_data[i:i+self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(frame)) != 0)
            zcr = zero_crossings / len(frame)
            speech_frames.append(zcr > threshold)
        
        return speech_frames
    
    def spectral_centroid_vad(self, threshold: float = 0.3) -> List[bool]:
        """Spectral centroid VAD"""
        print("🔍 Testing Spectral Centroid VAD...")
        
        if not SCIPY_AVAILABLE:
            print("❌ Skipping spectral centroid - scipy not available")
            return []
        
        speech_frames = []
        for i in range(0, len(self.audio_data), self.frame_size):
            frame = self.audio_data[i:i+self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            
            # Apply window
            windowed = frame * np.hanning(len(frame))
            
            # FFT
            fft = np.fft.fft(windowed)
            magnitude = np.abs(fft[:len(fft)//2])
            frequencies = np.fft.fftfreq(len(frame), 1/self.sample_rate)[:len(frame)//2]
            
            # Spectral centroid
            if np.sum(magnitude) > 0:
                centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
                # Normalize to 0-1 range
                normalized_centroid = centroid / (self.sample_rate / 2)
                speech_frames.append(normalized_centroid > threshold)
            else:
                speech_frames.append(False)
        
        return speech_frames
    
    def webrtc_vad(self, aggressiveness: int = 2) -> List[bool]:
        """WebRTC VAD"""
        if not WEBRTCVAD_AVAILABLE:
            print("❌ Skipping WebRTC VAD - webrtcvad not available")
            return []
        
        print("🔍 Testing WebRTC VAD...")
        
        # WebRTC VAD expects 16kHz, 8kHz, or 32kHz
        if self.sample_rate not in [8000, 16000, 32000]:
            print(f"⚠️  Resampling from {self.sample_rate}Hz to 16kHz for WebRTC VAD")
            # Simple resampling (for production, use proper resampling)
            resampled = signal.resample(self.audio_data, int(len(self.audio_data) * 16000 / self.sample_rate))
            sample_rate = 16000
        else:
            resampled = self.audio_data
            sample_rate = self.sample_rate
        
        # Convert to 16-bit PCM
        audio_int16 = (resampled * 32767).astype(np.int16)
        
        vad = webrtcvad.Vad(aggressiveness)
        speech_frames = []
        
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        for i in range(0, len(audio_int16), frame_size):
            frame = audio_int16[i:i+frame_size]
            if len(frame) == frame_size:
                is_speech = vad.is_speech(frame.tobytes(), sample_rate)
                speech_frames.append(is_speech)
        
        return speech_frames
    
    def torch_vad(self) -> List[bool]:
        """PyTorch-based VAD using torchaudio"""
        if not TORCH_VAD_AVAILABLE:
            print("❌ Skipping PyTorch VAD - torch/torchaudio not available")
            return []
        
        print("🔍 Testing PyTorch VAD...")
        
        # Load audio with torchaudio
        waveform, sample_rate = torchaudio.load(self.audio_file)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Apply VAD using torchaudio's VAD
        vad = torchaudio.transforms.Vad(sample_rate=sample_rate)
        
        speech_frames = []
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        for i in range(0, len(waveform), frame_size):
            frame = waveform[i:i+frame_size]
            if len(frame) >= frame_size:
                # Pad if needed
                if len(frame) < frame_size:
                    frame = torch.nn.functional.pad(frame, (0, frame_size - len(frame)))
                
                # Apply VAD
                is_speech = vad(frame, sample_rate)
                speech_frames.append(is_speech.item())
        
        return speech_frames
    
    def adaptive_threshold_vad(self, window_size: int = 10) -> List[bool]:
        """Adaptive threshold VAD"""
        print("🔍 Testing Adaptive Threshold VAD...")
        
        # Calculate RMS for all frames
        rms_values = []
        for i in range(0, len(self.audio_data), self.frame_size):
            frame = self.audio_data[i:i+self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            
            rms = np.sqrt(np.mean(frame**2))
            rms_values.append(rms)
        
        # Calculate adaptive threshold using sliding window
        speech_frames = []
        for i, rms in enumerate(rms_values):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(rms_values), i + window_size // 2 + 1)
            
            # Calculate local mean and std
            local_rms = rms_values[start_idx:end_idx]
            local_mean = np.mean(local_rms)
            local_std = np.std(local_rms)
            
            # Adaptive threshold
            threshold = local_mean + 0.5 * local_std
            speech_frames.append(rms > threshold)
        
        return speech_frames
    
    def run_all_tests(self) -> Dict:
        """Run all VAD algorithms and return results"""
        print("\n🧪 Running VAD Algorithm Comparison")
        print("=" * 50)
        
        algorithms = {
            "RMS-Very-Tolerant": lambda: self.rms_vad(threshold=0.03),
            "RMS-Tolerant": lambda: self.rms_vad(threshold=0.02),
            "RMS": lambda: self.rms_vad(threshold=0.01),
            "RMS-Aggressive": lambda: self.rms_vad(threshold=0.005),
            "RMS-Very-Aggressive": lambda: self.rms_vad(threshold=0.002),
            "Energy": lambda: self.energy_vad(threshold=0.005),
            "Zero-Crossing": lambda: self.zero_crossing_vad(threshold=0.1),
            "Spectral Centroid": lambda: self.spectral_centroid_vad(threshold=0.3),
            "Adaptive Threshold": lambda: self.adaptive_threshold_vad(window_size=10),
        }
        
        # Add optional algorithms
        if WEBRTCVAD_AVAILABLE:
            algorithms["WebRTC-Very-Tolerant"] = lambda: self.webrtc_vad(aggressiveness=0)  # Least aggressive
            algorithms["WebRTC-Tolerant"] = lambda: self.webrtc_vad(aggressiveness=1)   # Less aggressive
            algorithms["WebRTC"] = lambda: self.webrtc_vad(aggressiveness=2)
            algorithms["WebRTC-Aggressive"] = lambda: self.webrtc_vad(aggressiveness=3)  # Most aggressive
        
        if TORCH_VAD_AVAILABLE:
            algorithms["PyTorch"] = lambda: self.torch_vad()
        
        results = {}
        
        for name, algorithm in algorithms.items():
            try:
                start_time = time.time()
                speech_frames = algorithm()
                end_time = time.time()
                
                if speech_frames:
                    results[name] = {
                        "speech_frames": speech_frames,
                        "execution_time": end_time - start_time,
                        "speech_ratio": np.mean(speech_frames),
                        "total_frames": len(speech_frames)
                    }
                    print(f"✅ {name}: {len(speech_frames)} frames, {np.mean(speech_frames)*100:.1f}% speech, {end_time-start_time:.3f}s")
                else:
                    print(f"❌ {name}: Failed or returned empty result")
                    
            except Exception as e:
                print(f"❌ {name}: Error - {e}")
        
        self.results = results
        return results
    
    def print_detailed_results(self):
        """Print detailed analysis of results, including silence/speech transitions in ms"""
        print("\n📊 Detailed VAD Results")
        print("=" * 50)
        
        for name, result in self.results.items():
            speech_frames = result["speech_frames"]
            speech_ratio = result["speech_ratio"]
            total_frames = result["total_frames"]
            execution_time = result["execution_time"]
            
            print(f"\n🔍 {name}:")
            print(f"   Total frames: {total_frames}")
            print(f"   Speech frames: {np.sum(speech_frames)}")
            print(f"   Speech ratio: {speech_ratio*100:.1f}%")
            print(f"   Execution time: {execution_time:.3f}s")
            
            # Find speech segments
            speech_segments = self._find_speech_segments(speech_frames)
            print(f"   Speech segments: {len(speech_segments)}")
            
            for i, (start, end) in enumerate(speech_segments):  # Show all segments
                start_time = start * self.frame_duration_ms / 1000
                end_time = end * self.frame_duration_ms / 1000
                duration = end_time - start_time
                print(f"     Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")

            # Print transition times in ms
            print("   Transitions (ms):")
            last_state = speech_frames[0]
            print(f"     0: {'speech' if last_state else 'silence'}")
            for idx, state in enumerate(speech_frames[1:], 1):
                if state != last_state:
                    t_ms = idx * self.frame_duration_ms
                    print(f"     {t_ms}: {'speech' if state else 'silence'}")
                    last_state = state
    
    def _find_speech_segments(self, speech_frames: List[bool]) -> List[Tuple[int, int]]:
        """Find continuous speech segments"""
        segments = []
        start = None
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and start is None:
                start = i
            elif not is_speech and start is not None:
                segments.append((start, i))
                start = None
        
        # Handle case where speech continues to end
        if start is not None:
            segments.append((start, len(speech_frames)))
        
        return segments
    
    def save_results(self, output_file: str = "vad_results.json"):
        """Save results to JSON file"""
        output_data = {
            "audio_file": self.audio_file,
            "sample_rate": self.sample_rate,
            "frame_duration_ms": self.frame_duration_ms,
            "results": {}
        }
        
        for name, result in self.results.items():
            output_data["results"][name] = {
                "speech_ratio": result["speech_ratio"],
                "total_frames": result["total_frames"],
                "execution_time": result["execution_time"],
                "speech_segments": self._find_speech_segments(result["speech_frames"])
            }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to {output_file}")
    
    def create_cut_audio_files(self):
        """Create cut audio files that end at the final silence for each VAD algorithm"""
        print("\n✂️  Creating cut audio files...")
        
        for name, result in self.results.items():
            speech_frames = result["speech_frames"]
            
            # Find the last speech frame
            last_speech_frame = -1
            for i in range(len(speech_frames) - 1, -1, -1):
                if speech_frames[i]:
                    last_speech_frame = i
                    break
            
            if last_speech_frame >= 0:
                # Calculate the end sample (end of the last speech frame)
                end_frame = last_speech_frame + 1  # Include the full frame
                end_sample = end_frame * self.frame_size
                
                # Ensure we don't exceed the audio length
                end_sample = min(end_sample, len(self.audio_data))
                
                # Cut the audio
                cut_audio = self.audio_data[:end_sample]
                
                # Save the cut audio
                cut_filename = self.cut_dir / f"{name.replace(' ', '_').replace('-', '_')}_cut.wav"
                sf.write(str(cut_filename), cut_audio, self.sample_rate)
                
                cut_duration = len(cut_audio) / self.sample_rate
                original_duration = len(self.audio_data) / self.sample_rate
                print(f"   {name}: {cut_duration:.3f}s (cut from {original_duration:.3f}s)")
            else:
                print(f"   {name}: No speech detected, no cut file created")

def main():
    """Main function"""
    audio_file = "debug_wav_files/test_VAD.wav"
    
    if not Path(audio_file).exists():
        print(f"❌ Audio file {audio_file} not found!")
        return
    
    # Create VAD tester
    tester = VADTester(audio_file)
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Print detailed results
    tester.print_detailed_results()
    
    # Save results
    tester.save_results()
    
    # Create cut audio files
    tester.create_cut_audio_files()
    
    # Print total file duration
    total_duration_sec = len(tester.audio_data) / tester.sample_rate
    print(f"\n🔎 Total file duration: {total_duration_sec:.3f} seconds ({total_duration_sec*1000:.1f} ms)")
    
    print("\n✅ VAD testing completed!")

if __name__ == "__main__":
    main() 