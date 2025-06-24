export interface OrderItem {
  id: string;
  name: string;
  price: number;
  quantity: number;
}

export interface Order {
  items: OrderItem[];
  total: number;
}

export interface AudioMessage {
  type: 'transcription' | 'audio_response' | 'error';
  text?: string;
  message?: string;
}