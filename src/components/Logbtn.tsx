import { useEffect, useState } from "react";

const LogoutPage = () => {
  const [countdown, setCountdown] = useState(5);

  useEffect(() => {
    // Simulate logout process
    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-green-50">
      <div className="w-full max-w-md p-8 bg-white rounded-lg shadow-lg">
        <div className="text-center">
          <div className="mb-6">
            <svg
              className="mx-auto h-16 w-16 text-green-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M5 13l4 4L19 7"
              ></path>
            </svg>
          </div>

          <h1 className="text-2xl font-bold text-gray-800 mb-2">
            Successfully Logged Out
          </h1>
          <p className="text-gray-600 mb-6">
            Thank you for using the AgriTech Platform. Your session has been
            securely terminated.
          </p>

          <div className="mb-6 p-4 bg-green-50 rounded-md border border-green-100">
            <p className="text-sm text-green-800">
              For security reasons, please close your browser if you're using a
              shared device.
            </p>
          </div>

          <div className="space-y-4">
            <button
              onClick={() => (window.location.href = "/login")}
              className="w-full py-2 px-4 bg-green-600 hover:bg-green-700 text-white rounded-md font-medium transition duration-200"
            >
              Log In Again
            </button>

            <a
              href="/"
              className="block w-full py-2 px-4 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-md font-medium text-center transition duration-200"
            >
              Return to Homepage
            </a>

            {countdown > 0 && (
              <p className="text-sm text-gray-500">
                Redirecting to homepage in {countdown} second
                {countdown !== 1 ? "s" : ""}...
              </p>
            )}
          </div>
        </div>
      </div>

      <div className="mt-8 text-center text-sm text-gray-500">
        <p>
          &copy; {new Date().getFullYear()} AgriTech Platform. All rights
          reserved.
        </p>
        <div className="mt-2">
          <a href="/help" className="text-green-600 hover:text-green-800 mx-2">
            Help
          </a>
          <a
            href="/contact"
            className="text-green-600 hover:text-green-800 mx-2"
          >
            Contact
          </a>
          <a
            href="/privacy"
            className="text-green-600 hover:text-green-800 mx-2"
          >
            Privacy Policy
          </a>
        </div>
      </div>
    </div>
  );
};

export default LogoutPage;
