import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AquaPredict | Purifier care centre",
  description: "Synthetic filter-life forecasting dashboard for connected purifiers.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
