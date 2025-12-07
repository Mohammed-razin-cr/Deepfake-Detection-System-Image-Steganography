# Image Encryption/Decryption (Steganography) Feature

## ✅ Feature Added Successfully!

Your deepfake detection system now includes **image steganography** - the ability to hide and extract data from images!

## What is Steganography?

Steganography is the practice of hiding secret data within an ordinary file, message, or image. Unlike encryption which scrambles data, steganography hides data so that it's not visible to the naked eye.

## Features

### 1. **Encrypt Image** 🔒
- Hide messages, links, or any text data inside an image
- Data is hidden using LSB (Least Significant Bit) technique
- Original image looks unchanged
- Supports PNG, JPG, JPEG formats

### 2. **Decrypt Image** 🔓
- Extract hidden data from encrypted images
- Automatically detects if image contains hidden data
- Displays extracted message, link, or data

### 3. **Check if Encrypted** 🔍
- Quickly check if an image contains hidden data
- Shows available capacity for encryption

## How to Use

### Encrypt (Hide Data in Image)

1. Click the **"Encrypt Image"** tab
2. Upload an image (PNG, JPG, JPEG)
3. Enter the message/data you want to hide
4. Click **"Encrypt Image"**
5. Download the encrypted image

### Decrypt (Extract Data from Image)

1. Click the **"Decrypt Image"** tab
2. Upload an encrypted image
3. Click **"Decrypt Image"**
4. View the extracted data/link/message

### Check if Image is Encrypted

1. Click the **"Decrypt Image"** tab
2. Upload an image
3. Click **"Check if Encrypted"**
4. See if the image contains hidden data

## Technical Details

### Implementation

- **Method**: LSB (Least Significant Bit) Steganography
- **Storage**: Data is hidden in the least significant bits of RGB pixels
- **Format**: Encrypted images are saved as PNG for maximum compatibility
- **Capacity**: Depends on image size (typically 1/8 of total pixels)

### API Endpoints

- `POST /api/encrypt/image` - Encrypt (hide data) in image
- `POST /api/decrypt/image` - Decrypt (extract data) from image
- `POST /api/check/encrypted` - Check if image contains encrypted data

### Files Created

- `web_app/steganography.py` - Steganography utility class
- Updated `web_app/app.py` - Added encryption/decryption routes
- Updated `web_app/templates/index.html` - Added UI tabs
- Updated `web_app/static/script.js` - Added frontend functions

## Example Use Cases

1. **Hide Links**: Embed URLs or download links in images
2. **Secret Messages**: Send hidden messages in image files
3. **Data Storage**: Store text data invisibly in images
4. **Authentication**: Verify image authenticity with hidden metadata

## Security Notes

- ⚠️ Steganography hides data but doesn't encrypt it
- ⚠️ For sensitive data, encrypt first, then hide
- ⚠️ Always use trusted sources for encrypted images
- ⚠️ Large messages may be detectable due to file size changes

## Capacity Limits

The maximum data you can hide depends on:
- Image dimensions (width × height)
- Image format (PNG recommended)
- Typically: ~12.5% of total pixels

Example: A 1000×1000 image can hide approximately 125,000 characters.

## Testing

1. **Test Encryption**:
   - Upload an image
   - Enter a message: "This is a secret message!"
   - Encrypt and download

2. **Test Decryption**:
   - Upload the encrypted image
   - Click "Decrypt Image"
   - Verify the message appears

3. **Test Link Hiding**:
   - Encrypt a URL: "https://example.com"
   - Decrypt and click the link

## Troubleshooting

### "Message too long" error
- **Solution**: Use a larger image or shorter message

### "No encrypted data found"
- **Solution**: Image may not be encrypted or was corrupted

### Image looks different after encryption
- **Note**: This is normal - slight color changes occur but are minimal

---

## 🎉 Everything is Ready!

Your steganography feature is fully integrated and ready to use! Just start the server and try encrypting/decrypting images!

