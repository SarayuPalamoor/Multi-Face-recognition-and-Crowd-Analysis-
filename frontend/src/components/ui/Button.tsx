import React from 'react';
import '../../styles/Button.css';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: 'primary';
}

export const Button = ({ children, variant = 'primary', ...props }: ButtonProps) => {
  return (
    <button className="btn-12" {...props}>
      <span>{children}</span>
    </button>
  );
};