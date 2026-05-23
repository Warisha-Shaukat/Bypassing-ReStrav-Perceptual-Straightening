import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from scipy.optimize import minimize
import pywt
from torch.fft import fft2, ifft2, fftshift, ifftshift

# PyTorch CUDA 12 Pipeline - Often faster and simpler than CuPy

class TorchWaveletOptimizer:
    """PyTorch-based Wavelet Transform"""
    @staticmethod
    def apply_wavelet_denoising(channel_tensor, wavelet='db2', level=2):
        """
        Hybrid approach: CPU wavelets + GPU processing
        """
        device = channel_tensor.device
        
        # Convert to numpy for wavelet (PyTorch doesn't have native wavelet)
        channel_np = channel_tensor.cpu().detach().numpy()
        coeffs = pywt.wavedec2(channel_np, wavelet, level=level)
        coeffs_list = list(coeffs)
        
        # Soft-thresholding using PyTorch on GPU
        for i in range(1, len(coeffs_list)):
            processed_tuple = []
            for coeff in coeffs_list[i]:
                coeff_t = torch.from_numpy(coeff).to(device).float()
                threshold = torch.std(coeff_t) * 0.5
                soft_thresh = torch.sign(coeff_t) * torch.clamp(
                    torch.abs(coeff_t) - threshold, min=0
                )
                processed_tuple.append(soft_thresh.cpu().numpy())
            coeffs_list[i] = tuple(processed_tuple)
        
        result = pywt.waverec2(coeffs_list, wavelet)
        return torch.from_numpy(result).to(device).float()


class TorchFFTSmoother:
    """PyTorch FFT-based spectral smoothing"""
    @staticmethod
    def apply_spectral_smoothing(channel_tensor):
        """
        Butterworth low-pass filter using PyTorch FFT
        """
        device = channel_tensor.device
        
        # Ensure channel_tensor is 2D
        if channel_tensor.dim() == 3:
            channel_tensor = channel_tensor[0]
        
        # FFT
        f = fft2(channel_tensor)
        fshift = fftshift(f)
        
        rows, cols = channel_tensor.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create Butterworth mask on GPU
        y = torch.arange(-crow, rows - crow, dtype=torch.float32, device=device)
        x = torch.arange(-ccol, cols - ccol, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        radius = rows / 10.0
        mask = 1.0 / (1.0 + (torch.sqrt(xx**2 + yy**2) / radius) ** 2)
        
        fshift_filtered = fshift * mask
        f_ishift = ifftshift(fshift_filtered)
        img_back = ifft2(f_ishift)
        
        return torch.abs(img_back)


class TorchAdversarialEngine:
    """PyTorch-based adversarial optimization"""
    def __init__(self, device='cuda'):
        self.device = device
    
    @staticmethod
    def calculate_loss(perturbed_frame):
        """Entropy maximization"""
        # Use log-sum-exp for numerical stability
        log_prob = torch.log2(torch.abs(perturbed_frame) + 1e-7)
        entropy = -torch.sum(perturbed_frame * log_prob)
        return entropy
    
    def optimize_frame(self, frame_tensor):
        """
        Optimized noise pattern using adversarial attack.
        Uses PyTorch autograd for gradient computation.
        """
        frame_tensor = frame_tensor.detach().clone()
        original_shape = frame_tensor.shape
        
        # Initialize perturbation
        perturbation = torch.randn_like(frame_tensor) * 0.05
        perturbation = perturbation.to(self.device)
        perturbation.requires_grad = True
        
        # Optimizer
        optimizer = torch.optim.LBFGS([perturbation], lr=0.1, max_iter=5)
        
        def closure():
            optimizer.zero_grad()
            perturbed = frame_tensor + perturbation
            loss = -self.calculate_loss(perturbed)  # Negative because we maximize entropy
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return frame_tensor + perturbation.detach()


class TorchSPNEngine:
    """Sensor Pattern Noise using PyTorch tensors"""
    def __init__(self, shape, device='cuda'):
        self.device = device
        self.pattern = torch.randn(shape, dtype=torch.float32, device=device) * 0.02
    
    def apply_spn(self, frame_tensor):
        return frame_tensor + self.pattern


class HighFP_Pipeline_Torch:
    """
    PyTorch CUDA 12 optimized video processing pipeline.
    
    Advantages over CuPy:
    - Better GPU integration
    - Automatic differentiation for optimization
    - More optimized kernels
    - Easier debugging
    """
    def __init__(self, input_dir, output_dir, batch_size=8, device='cuda'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = device
        
        # Check GPU availability
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        self.wavelet_tool = TorchWaveletOptimizer()
        self.fft_tool = TorchFFTSmoother()
        self.adv_engine = TorchAdversarialEngine(device=self.device)
        self.spn_engine = None
        
        # GPU info
        if self.device == 'cuda':
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        os.makedirs(self.output_dir, exist_ok=True)

    def process_frame_gpu(self, frame):
        """Process single frame on GPU"""
        # Convert to tensor and transfer to GPU
        frame_tensor = torch.from_numpy(frame).float().to(self.device)
        
        # Initialize SPN
        if self.spn_engine is None:
            self.spn_engine = TorchSPNEngine(frame_tensor.shape, device=self.device)
        
        # 1. SPN Injection
        frame_tensor = self.spn_engine.apply_spn(frame_tensor)
        
        processed_channels = []
        for c in range(3):
            ch = frame_tensor[:, :, c]
            
            # 2. Wavelet Denoising
            ch = self.wavelet_tool.apply_wavelet_denoising(ch)
            
            # 3. FFT Spectral Smoothing
            ch = self.fft_tool.apply_spectral_smoothing(ch)
            
            # 4. Adversarial Optimization
            ch_adv = self.adv_engine.optimize_frame(ch)
            processed_channels.append(ch_adv)
        
        # Stack and transfer back to CPU
        result = torch.stack(processed_channels, dim=2)
        return result.cpu().numpy()

    def process_frames_batch_gpu(self, frames):
        """Process multiple frames in batches"""
        processed = []
        
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            
            # Convert batch to tensor
            batch_tensor = torch.from_numpy(np.array(batch)).float().to(self.device)
            
            # Initialize or refresh SPN when video/frame shape changes
            batch_shape = batch_tensor[0].shape
            if self.spn_engine is None or self.spn_engine.pattern.shape != batch_shape:
                self.spn_engine = TorchSPNEngine(batch_shape, device=self.device)
            
            # Process each frame
            for j in range(len(batch_tensor)):
                frame_tensor = batch_tensor[j]
                
                # Refresh SPN if this frame has different shape than current pattern
                if frame_tensor.shape != self.spn_engine.pattern.shape:
                    self.spn_engine = TorchSPNEngine(frame_tensor.shape, device=self.device)
                
                # Apply SPN
                frame_tensor = self.spn_engine.apply_spn(frame_tensor)
                
                processed_channels = []
                for c in range(3):
                    ch = frame_tensor[:, :, c]
                    ch = self.wavelet_tool.apply_wavelet_denoising(ch)
                    ch = self.fft_tool.apply_spectral_smoothing(ch)
                    ch_adv = self.adv_engine.optimize_frame(ch)
                    processed_channels.append(ch_adv)
                
                result = torch.stack(processed_channels, dim=2)
                processed.append(result.cpu().numpy())
        
        return processed

    def temporal_smooth_gpu(self, video_data):
        """GPU-accelerated temporal smoothing using Gaussian blur"""
        # Convert to tensor
        video_tensor = torch.from_numpy(video_data).float().to(self.device)
        
        # Reshape for 3D convolution: (frames, channels, height, width)
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        
        # Apply temporal smoothing using 1D convolution in time dimension
        smoothed = torch.zeros_like(video_tensor)
        for c in range(video_tensor.shape[1]):
            channel = video_tensor[:, c:c+1, :, :]  # (T, 1, H, W)
            
            # Create Gaussian kernel for temporal smoothing
            kernel_size = 5
            sigma = 0.5
            kernel = self._create_temporal_kernel(kernel_size, sigma, device=self.device)
            
            # Manually pad temporal dimension by reflecting edge frames
            pad_size = kernel_size // 2
            padded = torch.cat([torch.flip(channel[:pad_size], [0]), channel, torch.flip(channel[-pad_size:], [0])], dim=0)
            
            # Simple temporal smoothing
            for t in range(channel.shape[0]):
                window = padded[t:t+kernel_size, :, :, :]  # Slice temporal dimension
                smoothed[t, c:c+1, :, :] = window.mean(dim=0, keepdim=True)  # Average across time
        
        # Permute back to (T, H, W, C)
        smoothed = smoothed.permute(0, 2, 3, 1)
        
        return smoothed.cpu().numpy()

    @staticmethod
    def _create_temporal_kernel(size, sigma, device='cuda'):
        """Create 1D Gaussian kernel"""
        x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        return kernel / kernel.sum()

    def run_pipeline(self):
        """Main pipeline execution"""
        if not os.path.exists(self.input_dir):
            print(f"Error: Input directory '{self.input_dir}' not found.")
            return

        all_files = [f for f in os.listdir(self.input_dir) 
                    if f.lower().endswith((".mp4", ".mov", ".avi"))]
        all_files.sort()

        for filename in all_files:
            in_p = os.path.join(self.input_dir, filename)
            out_p = os.path.join(self.output_dir, filename)

            if os.path.exists(out_p):
                print(f"⏭️  Skipping {filename}")
                continue

            print(f"\n🔬 GPU Processing (PyTorch): {filename}")
            
            # Read video
            cap = cv2.VideoCapture(in_p)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frames = []
            frame_count = 0
            print(f"  📹 Reading frames...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"     {frame_count} frames read")
            cap.release()

            if not frames:
                print(f"Warning: No frames read")
                continue

            print(f"  🖥️  Processing {len(frames)} frames with GPU...")
            processed_frames = self.process_frames_batch_gpu(frames)

            print(f"  🌊 Temporal smoothing...")
            video_data = np.array(processed_frames, dtype=np.float32)
            smoothed_video = self.temporal_smooth_gpu(video_data)

            print(f"  💾 Writing video...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_p, fourcc, fps, (w, h))
            for f in smoothed_video:
                out.write(np.clip(f, 0, 255).astype(np.uint8))
            out.release()
            
            print(f"✅ Success: {filename}")


if __name__ == "__main__":
    INPUT = "test_fake"
    OUTPUT = "test_repaired2"
    
    # PyTorch GPU pipeline (recommended for most users)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = HighFP_Pipeline_Torch(
        INPUT, 
        OUTPUT, 
        batch_size=8,
        device=device
    )
    pipeline.run_pipeline()