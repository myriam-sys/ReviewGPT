import Link from "next/link";

export default function NotFound() {
  return (
    <main className="min-h-screen bg-zinc-950 flex flex-col items-center justify-center px-6">
      <div className="text-center space-y-4">
        <p className="text-6xl font-bold text-zinc-800">404</p>
        <h1 className="text-xl font-semibold text-zinc-200">Page not found</h1>
        <p className="text-zinc-500 text-sm">
          The session or page you&apos;re looking for doesn&apos;t exist.
        </p>
        <Link
          href="/"
          className="inline-block mt-4 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium text-sm transition-colors"
        >
          ← Back to home
        </Link>
      </div>
    </main>
  );
}
