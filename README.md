# Linear Regression Model with Rust and Burn

## Introduction
This project implements a linear regression model to predict \( y = 2x + 1 \) using the Rust Burn library (v0.16.0).

## Setup
1. Open this project in Replit: [replit.com](https://replit.com).
2. Run `cargo run` in the terminal to execute the code.

## Approach
- Generated synthetic data with noise using `rand`.
- Defined a linear regression model with `burn::nn::Linear`.
- Trained the model using SGD and MSE loss.
- Plotted results with `textplots`.

## Results
The model converges to predict \( y \approx 2x + 1 \), with a final loss < 0.01. The text plot shows a near-linear trend.

## Reflection
I used Replit to avoid local setup and relied on Burn’s docs and AI assistance for syntax. The main challenge was tensor reshaping, resolved by checking the library’s examples.

## Resources
- [Burn Docs](https://docs.rs/burn/0.16.0/burn/)
- [Rust Docs](https://doc.rust-lang.org/)
