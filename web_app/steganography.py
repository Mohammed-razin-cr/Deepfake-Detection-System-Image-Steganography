"""Image steganography utility for hiding and extracting data from images."""

import os
from PIL import Image
import base64
import json
from io import BytesIO


class ImageSteganography:
    """Handle image steganography operations (encrypt/decrypt data in images)."""
    
    def __init__(self):
        self.MAX_DATA_SIZE_MB = 5  # Maximum 5MB of data
    
    def _text_to_binary(self, text):
        """Convert text to binary string."""
        if isinstance(text, str):
            return ''.join(format(ord(char), '08b') for char in text)
        return text
    
    def _binary_to_text(self, binary):
        """Convert binary string to text."""
        chars = []
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                chars.append(chr(int(byte, 2)))
        return ''.join(chars)
    
    def _encode_data(self, image_path, data, output_path=None):
        """
        Hide data in image using LSB (Least Significant Bit) steganography.
        
        Args:
            image_path: Path to the cover image
            data: Data to hide (text, link, etc.)
            output_path: Path to save the encoded image (optional)
        
        Returns:
            Path to the encoded image
        """
        try:
            # Open the image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Prepare data
            # Format: [LENGTH][DATA][END_MARKER]
            data_str = str(data)
            length = len(data_str)
            
            # Check if image can hold the data
            max_capacity = (image.width * image.height * 3) // 8  # 3 channels (RGB)
            if length > max_capacity - 100:  # Reserve space for metadata
                raise ValueError(f"Image too small to hide data. Maximum capacity: {max_capacity - 100} characters")
            
            # Convert data to binary
            length_binary = format(length, '032b')  # 32 bits for length
            data_binary = self._text_to_binary(data_str)
            end_marker = '1111111111111110'  # 16-bit end marker
            
            # Combine: length + data + end marker
            binary_data = length_binary + data_binary + end_marker
            
            # Get pixel data
            pixels = list(image.getdata())
            data_index = 0
            
            # Embed data in LSB of pixels
            new_pixels = []
            for pixel in pixels:
                if data_index < len(binary_data):
                    r, g, b = pixel
                    
                    # Modify LSB of each color channel
                    if data_index < len(binary_data):
                        r = (r & 0xFE) | int(binary_data[data_index])
                        data_index += 1
                    
                    if data_index < len(binary_data):
                        g = (g & 0xFE) | int(binary_data[data_index])
                        data_index += 1
                    
                    if data_index < len(binary_data):
                        b = (b & 0xFE) | int(binary_data[data_index])
                        data_index += 1
                    
                    new_pixels.append((r, g, b))
                else:
                    new_pixels.append(pixel)
            
            # Create new image with hidden data
            encoded_image = Image.new('RGB', image.size)
            encoded_image.putdata(new_pixels)
            
            # Save the encoded image
            if output_path is None:
                base_name = os.path.splitext(image_path)[0]
                output_path = f"{base_name}_encoded.png"
            
            encoded_image.save(output_path, 'PNG')
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Failed to encode data: {str(e)}")
    
    def _decode_data(self, image_path):
        """
        Extract hidden data from image.
        
        Args:
            image_path: Path to the encoded image
        
        Returns:
            Extracted data string, or None if no data found
        """
        try:
            # Open the image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get pixel data
            pixels = list(image.getdata())
            
            # Extract binary data from LSB
            binary_data = ''
            for pixel in pixels:
                r, g, b = pixel
                binary_data += str(r & 1)
                binary_data += str(g & 1)
                binary_data += str(b & 1)
            
            # Find end marker
            end_marker = '1111111111111110'
            end_index = binary_data.find(end_marker)
            
            if end_index == -1:
                return None  # No data found
            
            # Extract length (first 32 bits)
            length_binary = binary_data[:32]
            length = int(length_binary, 2)
            
            # Extract data
            data_binary = binary_data[32:end_index]
            
            # Convert binary to text
            if len(data_binary) < length * 8:
                return None  # Invalid data
            
            extracted_data = self._binary_to_text(data_binary[:length * 8])
            
            return extracted_data
            
        except Exception as e:
            raise Exception(f"Failed to decode data: {str(e)}")
    
    def encrypt_image(self, image_path, message, output_path=None):
        """
        Encrypt (hide) message in image.
        
        Args:
            image_path: Path to cover image
            message: Message to hide (text, link, etc.)
            output_path: Path to save encrypted image
        
        Returns:
            Path to encrypted image
        """
        return self._encode_data(image_path, message, output_path)
    
    def decrypt_image(self, image_path):
        """
        Decrypt (extract) message from image.
        
        Args:
            image_path: Path to encrypted image
        
        Returns:
            Extracted message or None if no data found
        """
        return self._decode_data(image_path)
    
    def check_if_encrypted(self, image_path):
        """
        Check if image contains hidden data.
        
        Args:
            image_path: Path to image
        
        Returns:
            True if encrypted data found, False otherwise
        """
        try:
            data = self.decrypt_image(image_path)
            return data is not None and len(data) > 0
        except:
            return False
    
    def get_capacity(self, image_path):
        """
        Get maximum data capacity of an image.
        
        Args:
            image_path: Path to image
        
        Returns:
            Maximum characters that can be hidden
        """
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            max_capacity = (image.width * image.height * 3) // 8
            return max_capacity - 100  # Reserve space for metadata
        except:
            return 0

