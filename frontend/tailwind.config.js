/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        forest: {
          50: '#f2f9f4',
          100: '#e1f2e6',
          200: '#c4e5cf',
          300: '#97d1ac',
          400: '#64b684',
          500: '#3f9a63',
          600: '#2f7c4e',
          700: '#276240',
          800: '#224e35',
          900: '#1d412d',
          950: '#0b2316',
        },
        earth: {
          50: '#fbf8f4',
          100: '#f5efe6',
          200: '#ebdecd',
          300: '#ddc7ab',
          400: '#caa884',
          500: '#bc8f65',
          600: '#ae7c55',
          700: '#916346',
          800: '#76513c',
          900: '#604334',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      }
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
