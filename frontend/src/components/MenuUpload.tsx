import { useState } from 'react';

export function MenuUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<string>('');
  const [textContent, setTextContent] = useState<string>('');

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setUploadMessage('');
    }
  };

  const handleTextChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setTextContent(event.target.value);
  };

  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload-menu', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      
      if (response.ok) {
        setUploadMessage(`✅ ${result.message}`);
        setSelectedFile(null);
        // Reset file input
        const fileInput = document.getElementById('file-input') as HTMLInputElement;
        if (fileInput) fileInput.value = '';
      } else {
        setUploadMessage(`❌ Error: ${result.error}`);
      }
    } catch (error) {
      setUploadMessage(`❌ Upload failed: ${error}`);
    }
  };

  const uploadTextContent = async () => {
    if (!textContent.trim()) {
      setUploadMessage('❌ Please enter some text content');
      return;
    }

    // Create a text file from the content
    const blob = new Blob([textContent], { type: 'text/plain' });
    const file = new File([blob], `menu_${Date.now()}.txt`, { type: 'text/plain' });
    
    await uploadFile(file);
    setTextContent('');
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadMessage('❌ Please select a file first');
      return;
    }

    setIsUploading(true);
    setUploadMessage('');

    await uploadFile(selectedFile);
    setIsUploading(false);
  };

  return (
    <div>
      <h2 style={{ marginTop: 0, color: '#1f2937' }}>Menu Upload</h2>
      <p style={{ color: '#6b7280', marginBottom: '32px' }}>
        Upload your restaurant menu as text or image. This will be used to help the AI understand your offerings.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
        {/* Text Input Section */}
        <div style={{ padding: '24px', border: '1px solid #e5e7eb', borderRadius: '8px' }}>
          <h3 style={{ marginTop: 0, color: '#374151' }}>Enter Menu Text</h3>
          <textarea
            value={textContent}
            onChange={handleTextChange}
            placeholder="Enter your menu items here...

Example:
Cheeseburger - $8.99
Fries - $3.99
Drink - $2.99"
            style={{
              width: '100%',
              padding: '12px',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              fontFamily: 'inherit',
              fontSize: '14px',
              marginBottom: '16px',
              resize: 'vertical'
            }}
            rows={8}
          />
          <button
            onClick={uploadTextContent}
            disabled={isUploading || !textContent.trim()}
            style={{
              padding: '12px 24px',
              border: 'none',
              backgroundColor: (isUploading || !textContent.trim()) ? '#9ca3af' : '#3b82f6',
              color: 'white',
              borderRadius: '6px',
              cursor: (isUploading || !textContent.trim()) ? 'not-allowed' : 'pointer',
              fontWeight: '500'
            }}
          >
            {isUploading ? 'Uploading...' : 'Save Menu Text'}
          </button>
        </div>

        <div style={{ textAlign: 'center', color: '#6b7280', fontWeight: '500', margin: '16px 0' }}>
          OR
        </div>

        {/* File Upload Section */}
        <div style={{ padding: '24px', border: '1px solid #e5e7eb', borderRadius: '8px' }}>
          <h3 style={{ marginTop: 0, color: '#374151' }}>Upload Menu File</h3>
          <div style={{ marginBottom: '16px' }}>
            <input
              id="file-input"
              type="file"
              onChange={handleFileSelect}
              accept=".txt,.png,.jpg,.jpeg,.pdf"
              style={{ display: 'none' }}
            />
            <label 
              htmlFor="file-input" 
              style={{
                display: 'inline-block',
                padding: '12px 16px',
                backgroundColor: '#f9fafb',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                cursor: 'pointer'
              }}
            >
              {selectedFile ? selectedFile.name : 'Choose file...'}
            </label>
          </div>
          
          <div style={{ marginBottom: '16px' }}>
            <small style={{ color: '#6b7280' }}>
              Supported: Text files (.txt) and Images (.png, .jpg, .jpeg)
            </small>
          </div>

          <button
            onClick={handleUpload}
            disabled={isUploading || !selectedFile}
            style={{
              padding: '12px 24px',
              border: 'none',
              backgroundColor: (isUploading || !selectedFile) ? '#9ca3af' : '#3b82f6',
              color: 'white',
              borderRadius: '6px',
              cursor: (isUploading || !selectedFile) ? 'not-allowed' : 'pointer',
              fontWeight: '500'
            }}
          >
            {isUploading ? 'Uploading...' : 'Upload File'}
          </button>
        </div>
      </div>

      {uploadMessage && (
        <div style={{
          padding: '16px',
          borderRadius: '6px',
          marginTop: '16px',
          backgroundColor: uploadMessage.startsWith('✅') ? '#d1fae5' : '#fee2e2',
          color: uploadMessage.startsWith('✅') ? '#065f46' : '#991b1b',
          border: `1px solid ${uploadMessage.startsWith('✅') ? '#a7f3d0' : '#fca5a5'}`
        }}>
          {uploadMessage}
        </div>
      )}

      <div style={{ 
        marginTop: '32px', 
        padding: '24px', 
        backgroundColor: '#f9fafb', 
        borderRadius: '8px' 
      }}>
        <h3 style={{ marginTop: 0, color: '#374151' }}>How it works:</h3>
        <ul style={{ color: '#6b7280', margin: 0 }}>
          <li style={{ marginBottom: '8px' }}>Text files are saved directly to the menu database</li>
          <li style={{ marginBottom: '8px' }}>Images are processed with OCR to extract menu text (currently stubbed)</li>
          <li style={{ marginBottom: '8px' }}>The AI uses this menu information to understand your offerings when taking orders</li>
        </ul>
      </div>
    </div>
  );
}