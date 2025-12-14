import "./globals.css";

export const metadata = {
  title: "NepEd Chat",
  description: "Chat-style RAG assistant for NepEd",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
