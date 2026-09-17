import type { Metadata } from 'next';
import '@mobilerun/react/styles.css';
import './globals.css';

export const metadata: Metadata = {
  title: 'Mobile Agent',
  description: 'A live workspace for your mobile agent.',
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
