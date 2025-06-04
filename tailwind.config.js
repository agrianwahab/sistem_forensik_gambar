/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#2238A4', // Dark blue
          light: '#5AA8D6', // Light blue
          dark: '#1B2B80', // Darker blue
        },
        secondary: {
          DEFAULT: '#6c757d', // Gray
          light: '#f4f7f6', // Light gray background
          dark: '#333333', // Dark gray text
        }
      }
    },
  },
  plugins: [],
}