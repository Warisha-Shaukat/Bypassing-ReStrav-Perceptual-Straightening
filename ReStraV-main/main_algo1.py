import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os

class SpectralProcessor:
    """Handles Frequency Domain Manipulations (Fourier/DIP)"""
    @staticmethod
    def get_butterworth_mask(h, w, radius, order=2):
        """Creates a Butterworth Low-Pass Filter to suppress checkerboard artifacts."""
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist_sq = (X - center_x)**2 + (Y - center_y)**2
        # Butterworth formula: 1 / (1 + (D/D0)^(2n))
        mask = 1 / (1 + (np.sqrt(dist_sq) / radius)**(2 * order))
        return mask

    @staticmethod
    def repair_spectrum(channel, mask):
        """Applies frequency masking to normalize the Power Spectral Density."""
        # Fast Fourier Transform
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)

        # Apply mask to suppress high-frequency 'spikes'
        filtered_dft = dft_shift * mask

        # Inverse FFT
        idft_shift = np.fft.ifftshift(filtered_dft)
        img_back = np.fft.ifft2(idft_shift)
        return np.abs(img_back)

class NoiseEngine:
    """Simulates physical sensor characteristics."""
    
    @staticmethod
    def apply_sensor_noise(channel, sigma=1.2):
        """Injects Additive White Gaussian Noise (AWGN) to mask AI smoothness."""
        noise = np.random.normal(0, sigma, channel.shape)
        return channel + noise

class TemporalOptimizer:

    """Handles frame-to-frame coherence and motion smoothing."""

   

    @staticmethod

    def smooth_trajectory(frame_buffer, sigma=0.7):

        """

        Applies Gaussian smoothing across the temporal axis.

        This 'straightens' the neural curvature detected by ReStraV.

        """

        # axis=0 is the time/frame axis

        return gaussian_filter1d(frame_buffer, sigma=sigma, axis=0)



class AntiForensicsPipeline:

    def __init__(self, radius=60, noise_val=1.5, temp_sigma=0.5):

        self.radius = radius

        self.noise_val = noise_val

        self.temp_sigma = temp_sigma



    def process_video(self, input_path, output_path):

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():

            print(f"❌ Error: Could not open {input_path}")

            return



        # Video Metadata

        fps = cap.get(cv2.CAP_PROP_FPS)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

       

        # Initialize Spectral Mask

        mask = SpectralProcessor.get_butterworth_mask(h, w, self.radius)

       

        frames = []

        print(f"🔬 Analyzing & Filtering: {os.path.basename(input_path)}")



        while True:

            ret, frame = cap.read()

            if not ret:

                break



            # 1. Convert to float and process each RGB channel for spectral artifacts

            frame_float = frame.astype(np.float32)

           

            repaired_channels = []

            for c in range(3):  # R, G, B

                channel = frame_float[:, :, c]

               

                # 2. Frequency Domain Repair

                channel_repaired = SpectralProcessor.repair_spectrum(channel, mask)

               

                # 3. Inject Sensor Noise

                channel_noisy = NoiseEngine.apply_sensor_noise(channel_repaired, self.noise_val)

               

                repaired_channels.append(channel_noisy)

           

            # 4. Reconstruct Frame

            bgr_reconstructed = np.stack(repaired_channels, axis=2).astype(np.uint8)

            frames.append(bgr_reconstructed)



        cap.release()



        # 5. Temporal Smoothing (Anti-ReStraV)

        print("🕒 Correcting Temporal Curvature...")

        video_data = np.array(frames, dtype=np.float32)

        smoothed_video = TemporalOptimizer.smooth_trajectory(video_data, self.temp_sigma)



        # 6. Final Write

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

       

        for f in smoothed_video:

            out.write(np.clip(f, 0, 255).astype(np.uint8))

       

        out.release()

        print(f"✅ Repaired file saved to: {output_path}")



# --- Main Execution ---

if __name__ == "__main__":

    INPUT_DIR = "test_fake"

    OUTPUT_DIR = "test_repaired1"

    os.makedirs(OUTPUT_DIR, exist_ok=True)



    # Initialize researcher tool

    # radius: higher = more aggressive high-frequency suppression

    # noise_sigma: higher = more noise injection to mask AI patterns

    # temp_sigma: higher = stronger temporal smoothing to straighten trajectories

    engine = AntiForensicsPipeline(radius=100, noise_val=2.5, temp_sigma=2.0)

   



    for filename in os.listdir(INPUT_DIR):

        print(f"jfnfrj")

        if filename.lower().endswith((".mp4", ".mov", ".avi")):

            print(f"jfnfrj")

            in_p = os.path.join(INPUT_DIR, filename)

            out_p = os.path.join(OUTPUT_DIR, filename)

            engine.process_video(in_p, out_p)
