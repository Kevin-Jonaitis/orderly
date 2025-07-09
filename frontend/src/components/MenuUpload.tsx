import { useState } from 'react';
import { Button, Form, Alert, Card, Stack, Container, Spinner } from 'react-bootstrap';

export function MenuUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<string>('');

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setUploadMessage('');
    }
  };

  const uploadAndProcessMenu = async () => {
    if (!selectedFile) {
      setUploadMessage('❌ Please select an image file first');
      return;
    }

    setIsUploading(true);
    setIsProcessing(false);
    setUploadMessage('');

    try {
      // Step 1: Upload the image
      const formData = new FormData();
      formData.append('file', selectedFile);

      const uploadResponse = await fetch('http://localhost:8002/api/upload-menu', {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        const error = await uploadResponse.json();
        throw new Error(error.detail || 'Upload failed');
      }

      const uploadResult = await uploadResponse.json();
      setUploadMessage('📤 Image uploaded successfully. Processing with OCR...');

      // Step 2: Process the image with OCR
      setIsUploading(false);
      setIsProcessing(true);

      const processResponse = await fetch('http://localhost:8002/api/process-menu', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: "menu.jpg" }),
      });

      if (!processResponse.ok) {
        const error = await processResponse.json();
        throw new Error(error.detail || 'OCR processing failed');
      }

      setUploadMessage('✅ Menu processed successfully! The AI will now use the updated menu.');
      setSelectedFile(null);
      
      // Reset file input
      const fileInput = document.getElementById('file-input') as HTMLInputElement;
      if (fileInput) fileInput.value = '';

    } catch (error) {
      setUploadMessage(`❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);
      setIsProcessing(false);
    }
  };

  return (
    <Container>
      <Stack gap={4}>
        <Stack gap={1}>
          <Card.Title as="h4">Upload Menu Image</Card.Title>
          <Card.Text className="text-muted">
            Upload a photo of your restaurant menu. The AI will use OCR to extract menu items and prices.
          </Card.Text>
        </Stack>

        <Card>
          <Card.Body>
            <Card.Title>Upload Menu Image</Card.Title>
            <Form>
              <Form.Group className="mb-3">
                <Form.Control
                  id="file-input"
                  type="file"
                  onChange={handleFileSelect}
                  accept=".png,.jpg,.jpeg"
                />
                <Form.Text className="text-muted">
                  Supported: Images (.png, .jpg, .jpeg)
                </Form.Text>
              </Form.Group>
              <Button
                onClick={uploadAndProcessMenu}
                disabled={isUploading || isProcessing || !selectedFile}
                variant="primary"
              >
                {isUploading && <Spinner animation="border" size="sm" className="me-2" />}
                {isProcessing && <Spinner animation="border" size="sm" className="me-2" />}
                {isUploading ? 'Uploading...' : isProcessing ? 'Processing with OCR...' : 'Upload & Process Menu'}
              </Button>
            </Form>
          </Card.Body>
        </Card>

        {uploadMessage && (
          <Alert 
            variant={uploadMessage.startsWith('✅') ? 'success' : uploadMessage.startsWith('📤') ? 'info' : 'danger'}
          >
            {uploadMessage}
          </Alert>
        )}

        <Alert variant="info">
          <Alert.Heading>How it works:</Alert.Heading>
          <Stack as="ul" gap={1} className="mb-0">
            <Card.Text as="li">Upload a clear photo of your menu</Card.Text>
            <Card.Text as="li">OCR technology extracts menu items and prices</Card.Text>
            <Card.Text as="li">The AI uses this information when taking orders</Card.Text>
            <Card.Text as="li">Menu updates are applied immediately</Card.Text>
          </Stack>
        </Alert>
      </Stack>
    </Container>
  );
}