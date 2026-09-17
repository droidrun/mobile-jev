import type { Metadata } from 'next';
import '@mobilerun/react/styles.css';
import './globals.css';

export const metadata: Metadata = {
  title: 'Mobile Jev · mobilerun',
  description:
    'A mobile agent by Droidrun. TypeSafe’s Jev makes the decisions; mobilerun controls the Android phone.',
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
