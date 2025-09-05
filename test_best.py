#!/usr/bin/env python3
"""
BMACE - MIREX 2025 Audio Chord Estimation
Safe inference script for MIREX submission (force 22050 Hz, normalization restored)
"""

import os
import torch
import argparse
import numpy as np
import warnings
import librosa
from scipy.ndimage import median_filter
from utils import logger
from btc_model import BMACE
from utils.hparams import HParams

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logger.logging_verbosity(1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Large vocabulary chord mapping
root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
quality_list = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']

def idx2voca_chord():
    idx2voca_chord = ['N'] * 170
    idx2voca_chord[169] = 'N'
    idx2voca_chord[168] = 'X'
    for i in range(168):
        root = i // 14
        root = root_list[root]
        quality = i % 14
        quality = quality_list[quality]
        if i % 14 == 1:  # Major chords explicit
            chord = root + ':maj'
        else:
            chord = root + ':' + quality
        idx2voca_chord[i] = chord
    return idx2voca_chord

class BMACEEstimator:
    def __init__(self, 
                 ckpt_file_name='model/bmace_test_BMACE_True_4.pth.tar',
                 norm_file_name='normalization/bmace_test_BMACE_True_4_normalization.pt'):
        self.ckpt_file_name = ckpt_file_name
        self.norm_file_name = norm_file_name

        # ✅ 항상 large voca (170 코드) 사용
        self.idxtochord = idx2voca_chord()

        # Load configuration
        self.config = HParams.load("run_config.yaml")
        self.config.feature['large_voca'] = True
        self.config.model['num_chords'] = 170   # ✅ 고정

        # Load model and normalization
        self.model = self._load_model()
        self.mean, self.std = self._load_normalization()

    def _load_model(self):
        """Load BMACE model with flexible checkpoint handling"""
        model = BMACE(config=self.config.model).to(device)

        if os.path.isfile(self.ckpt_file_name):
            checkpoint = torch.load(self.ckpt_file_name, map_location=device)

            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v

            model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"BMACE model loaded from {self.ckpt_file_name}")
        else:
            logger.error(f"No checkpoint found at {self.ckpt_file_name}")
            raise FileNotFoundError(f"Model file not found: {self.ckpt_file_name}")

        model.eval()
        return model

    def _load_normalization(self):
        """Load normalization if available, otherwise fallback"""
        if os.path.isfile(self.norm_file_name):
            norm_data = torch.load(self.norm_file_name, map_location="cpu")
            mean, std = norm_data.get("mean", 0.0), norm_data.get("std", 1.0)
            logger.info(f"Loaded normalization from {self.norm_file_name}: mean={mean:.4f}, std={std:.4f}")
        else:
            mean, std = 0.0, 1.0
            logger.warning("Normalization file not found, using mean=0, std=1")
        return mean, std

    def extract_features(self, audio_file):
        """Extract CQT features from audio file with optimized parameters"""
        try:
            y, sr = librosa.load(audio_file, sr=22050)
            
            # Improved CQT with better frequency range
            fmin = librosa.note_to_hz('C2')  # Start from C2 (65.4 Hz)
            cqt = librosa.cqt(y,
                              sr=sr,
                              hop_length=self.config.feature['hop_length'],
                              n_bins=self.config.feature['n_bins'],
                              bins_per_octave=self.config.feature['bins_per_octave'],
                              fmin=fmin,
                              filter_scale=1.0)

            cqt_mag = np.abs(cqt)
            
            # Better amplitude to dB conversion with proper reference
            cqt_log = librosa.amplitude_to_db(cqt_mag, ref=np.max, top_db=80.0)
            
            # Apply median filtering to reduce noise
            cqt_log = median_filter(cqt_log, size=(1, 3))  # Smooth along time axis

            features = torch.FloatTensor(cqt_log).unsqueeze(0).to(device)
            return features
        except Exception as e:
            logger.error(f"Error extracting features from {audio_file}: {str(e)}")
            raise

    def predict_chords(self, audio_file):
        """Predict chord sequence from audio file with improved post-processing"""
        try:
            # 1. Feature 추출 및 정규화
            features = self.extract_features(audio_file)
            features = (features - self.mean) / self.std
            features = features.permute(0, 2, 1)  # (B, T, F)

            # 2. 모델 추론 - validation mode settings
            self.model.eval()  # Ensure eval mode
            with torch.no_grad():
                outputs = self.model(features, None)

            # 3. Enhanced output handling with secondary task
            if isinstance(outputs, (tuple, list)):
                prediction = outputs[0]  # Primary chord prediction
                
                # Check for secondary task output (sec_acc from training log)
                if len(outputs) >= 4 and outputs[3] is not None:
                    secondary_pred = outputs[3]
                    if isinstance(secondary_pred, torch.Tensor):
                        # Ensemble-like combination: 80% primary + 20% secondary
                        prediction = 0.8 * prediction + 0.2 * secondary_pred
            else:
                prediction = outputs

            if not isinstance(prediction, torch.Tensor):
                prediction = torch.as_tensor(prediction)

            # Apply softmax for better probability distribution
            if prediction.dim() > 1:
                prediction = torch.softmax(prediction, dim=-1)
                prediction = torch.argmax(prediction, dim=-1)

            # (B*T,) → (T,) 시퀀스로 변환
            pred_idx_seq = prediction.view(-1).cpu().numpy()
            
            # Apply enhanced temporal smoothing
            pred_idx_seq = self._apply_temporal_smoothing(pred_idx_seq, window_size=7)

            # 4. 프레임 단위 → 타임스탬프 변환
            chord_sequence = []
            hop_length = self.config.feature['hop_length']
            sr = 22050
            frame_duration = hop_length / sr

            for i, pred_idx in enumerate(pred_idx_seq):
                start_time = i * frame_duration
                end_time = (i + 1) * frame_duration
                chord_label = self.idxtochord[pred_idx]
                chord_sequence.append((start_time, end_time, chord_label))

            # 5. 연속된 같은 코드 merge with minimum duration
            merged_sequence = self._merge_consecutive_chords(chord_sequence)
            
            # 6. Apply minimum duration filter
            filtered_sequence = self._apply_minimum_duration(merged_sequence, min_duration=0.15)
            
            # 7. Apply musical knowledge constraints
            final_sequence = self._apply_musical_constraints(filtered_sequence)
            
            return final_sequence

        except Exception as e:
            logger.error(f"Error predicting chords: {str(e)}")
            raise

    def _apply_temporal_smoothing(self, pred_seq, window_size=5):
        """Apply median filtering to smooth temporal predictions"""
        if len(pred_seq) < window_size:
            return pred_seq
        
        # Apply median filter to reduce isolated predictions
        smoothed = median_filter(pred_seq.astype(np.float32), size=window_size)
        return smoothed.astype(np.int64)
    
    def _apply_minimum_duration(self, chord_sequence, min_duration=0.1):
        """Remove very short chord segments and merge with neighbors"""
        if not chord_sequence:
            return []
            
        filtered = []
        for start_time, end_time, chord_label in chord_sequence:
            duration = end_time - start_time
            if duration >= min_duration:
                filtered.append((start_time, end_time, chord_label))
            elif filtered:
                # Extend the previous chord to cover this short segment
                prev_start, prev_end, prev_chord = filtered[-1]
                filtered[-1] = (prev_start, end_time, prev_chord)
        
        return filtered

    def _merge_consecutive_chords(self, chord_sequence):
        if not chord_sequence:
            return []

        merged = []
        current_start, current_end, current_chord = chord_sequence[0]

        for start_time, end_time, chord_label in chord_sequence[1:]:
            if chord_label == current_chord:
                current_end = end_time
            else:
                merged.append((current_start, current_end, current_chord))
                current_start, current_end, current_chord = start_time, end_time, chord_label

        merged.append((current_start, current_end, current_chord))
        return merged

    def _apply_musical_constraints(self, chord_sequence):
        """Apply musical knowledge constraints to improve chord transitions"""
        if len(chord_sequence) < 2:
            return chord_sequence
            
        # Define unlikely transitions (to penalize)
        unlikely_transitions = [
            ('C:maj', 'C#:maj'), ('C#:maj', 'C:maj'),  # Chromatic jumps
            ('F:maj', 'F#:maj'), ('F#:maj', 'F:maj'),
            ('G:maj', 'G#:maj'), ('G#:maj', 'G:maj'),
            ('D:maj', 'D#:maj'), ('D#:maj', 'D:maj'),
            ('A:maj', 'A#:maj'), ('A#:maj', 'A:maj'),
            ('E:maj', 'F:maj'),  ('F:maj', 'E:maj'),   # Semitone jumps
        ]
        
        refined_sequence = []
        for i, (start_time, end_time, chord_label) in enumerate(chord_sequence):
            if i == 0:
                refined_sequence.append((start_time, end_time, chord_label))
                continue
                
            prev_chord = refined_sequence[-1][2]
            
            # Check for unlikely transitions and very short segments
            duration = end_time - start_time
            if (prev_chord, chord_label) in unlikely_transitions and duration < 0.5:
                # Keep the previous chord instead for short unlikely transitions
                prev_start, prev_end, prev_label = refined_sequence[-1]
                refined_sequence[-1] = (prev_start, end_time, prev_label)
            else:
                refined_sequence.append((start_time, end_time, chord_label))
                
        return refined_sequence

def write_mirex_format(chord_sequence, output_file):
    with open(output_file, 'w') as f:
        for start_time, end_time, chord_label in chord_sequence:
            f.write(f"{start_time:.6f} {end_time:.6f} {chord_label}\n")

def main():
    parser = argparse.ArgumentParser(description='BMACE - Audio Chord Estimation for MIREX 2025')
    parser.add_argument('--input', required=True, help='Input audio file (.wav)')
    parser.add_argument('--output', required=True, help='Output file (dummy, will be overridden)')
    parser.add_argument('--ckpt_file_name', type=str,
                       default='model/bmace_test_BMACE_True_4.pth.tar',
                       help='BMACE checkpoint file name')
    parser.add_argument('--norm_file_name', type=str,
                       default='normalization/bmace_test_BMACE_True_4_normalization.pt',
                       help='Normalization file name')
    args = parser.parse_args()

    # ✅ config 불러오기
    from utils.hparams import HParams
    config = HParams.load("run_config.yaml")

    # ✅ voca 고정 (170 chords)
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170
    logger.info(f"num_chords fixed to {config.model['num_chords']} (large voca only)")

    if not os.path.exists(args.input):
        logger.error(f"Input file {args.input} not found")
        return 1

    try:
        logger.info(f"Processing {args.input} with BMACE model...")
        estimator = BMACEEstimator(
            ckpt_file_name=args.ckpt_file_name,
            norm_file_name=args.norm_file_name
        )
        chord_sequence = estimator.predict_chords(args.input)

        # ✅ output 파일명 규칙 적용 (01.wav → 01.wav.txt)
        base_name = os.path.basename(args.input)
        output_name = base_name + ".txt"
        output_path = os.path.join(os.path.dirname(args.output), output_name)

        write_mirex_format(chord_sequence, output_path)

        logger.info(f"Results written to {output_path}")
        logger.info(f"Processed {len(chord_sequence)} chord segments")
        return 0
    except Exception as e:
        logger.error(f"Error processing {args.input}: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())