import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp


from PIL import Image, ImageOps
import cv2
import numpy as np
import pywt
import pywt.data

from scipy import signal, ndimage


#ar_im = np.array(im)
#ar_im = im.convert("L")
#plt.imshow(ar_im)
#print(ar_im)

#gr_2 = ImageOps.grayscale(im_2)
#gr = ImageOps.grayscale(im)
#scale = np.array(gr)
#scale_2 = np.array(gr_2)
#corr = signal.correlate2d(scale, scale_2)
#plt.imshow(corr)
#plt.show()
# Load the two images
#image1 = np.array(Image.open('/Users/santoshbhandari/Desktop/Speckal Analysis/image_1.tif').convert('L'))
#image2 = np.array(Image.open('/Users/santoshbhandari/Desktop/Speckal Analysis/image_6.tif').convert('L'))

#x1, y1 = 1024,768
#w, h = 384, 512
#image1_part = image1[y1:y1+h, x1:x1+w]
#print(image1_part)
#image2_part = image2[y1:y1+h, x1:x1+w]
#print(image2_part)



# Compute the cross-correlation
#correlation = signal.correlate2d(image1_part, image2_part, mode='same')
#pr = print(image1_part)
#max_pos = np.unravel_index(np.argmax(correlation), correlation.shape)
#print(max_pos)
#print(correlation)

#img_blur = cv2.GaussianBlur(correlation,(5,5), 0) # apply Gaussian smoothing
#cv2.imshow('Original Image', correlation)
#cv2.imshow('correlation', img_blur)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#corr = signal.correlate2d([[-1,2],[5,-3]], [[-1,-1],[-1,-1]])
#mat = ([[1,2],[3,4]])
#mat2 =([[2,1],[4,3]])

#res = np.dot(mat,mat2)
#print(res)
#print(corr)


#import matplotlib.pyplot as plt
#cmap = plt.cm._get_cmap('viridis')
#cmap = plt.cm._get_cmap('binary')
#plt.imshow(correlation, cmap=cmap, origin='lower')
#plt.colorbar()
#plt.show()

#import inspect
#from scipy.signal import correlate2d

#source_code = inspect.getsource(correlate2d)
#print(source_code)



#import inspect

#from PIL import Image, ImageDraw

#specify = inspect.getsource(ImageDraw)
#print(specify)

#import cv2
#import numpy as np

# Load two images
#img1 = cv2.imread('/Users/santoshbhandari/Desktop/Speckal Analysis/image_1.tif', cv2.IMREAD_GRAYSCALE)
#img2 = cv2.imread('/Users/santoshbhandari/Desktop/Speckal Analysis/image_6.tif', cv2.IMREAD_GRAYSCALE)

# Calculate the difference between the images
#diff = cv2.absdiff(img1, img2)

# Threshold the difference image
#thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Apply a median filter to remove noise
#median = cv2.medianBlur(thresh, 5)

# Display the result
#cv2.imshow('Difference', diff)
#cv2.imshow('Threshold', thresh)
#cv2.imshow('Median', median)
#cv2.waitKey(0)

# Calculate the time history of the speckle pattern

#x1, y1 = 1025, 768
#w,h = 5, 5

#image_part_1 = img1[y1:y1+h, x1:x1+w]
#image_part_2 = img2[y1:y1+h, x1:x1+w]
#history = np.zeros((image_part_1.shape[0], image_part_2.shape[1]))
#for i in range(img1.shape[0]):
 #   for j in range(img1.shape[1]):
  #      if median[i,j] == 255:
   #         history[i,j] += 2

# Display the time history
#cv2.imshow('History', history.astype(np.uint8))
#cv2.waitKey(0)

# Save the time history as an image
#cv2.imwrite('history.jpg', history)



#import numpy as np

#Defining the data into 4 by 4 matrix



# Define the factor loadings matrix
#factor_loadings = np.array([
   # [0.0, 0.0, 0.0, 0.0],
    #[0.7, 0.3, 0.0, 0.0],
    #[0.0, 0.0, 0.6, 0.4],
    #[0.0, 0.0, 0.5, 0.5]#
#])#

# Define the data matrix (with four variables and five observations)
#data = np.array([
 #   [2.5, 2.4, 0.5, 1.8],
 #   [0.5, 0.7, 1.2, 0.9],
 #   [2.2, 2.9, 2.4, 1.6],
 #   [1.9, 2.2, 3.0, 2.1],
 #   [3.1, 3.0, 2.7, 2.1]
#])

# Compute the factor scores
#factor_scores = np.dot(data, factor_loadings)

# Print the results
#print("Factor scores:")
#print(factor_scores)

from scipy.optimize import fsolve

# Define the system of equations
#def equations(variables):
    #x1, x2 = variables
   # eq1 = -2.24*x2 - 5.2*x1 + 9.6*x1*x2 + 1.08 - 0.125
  #  eq2 = -12*x2 - 50*x1 + 80*x1*x2 + 4
 #   return [eq1, eq2]

# Solve the system of equations
#x1, x2 = fsolve(equations, (0, 0))

# Print the solutions
#print("x1 =", x1)
#print("x2 =", x2)



#To simulate the convolution for a speckle pattern.
#let us first consider the electromagnetic wave which is useful in the place of the speckle which is for nbow explained as the "n points which is used in the array of the speckle pattern."
#"let us emit a light of radiation of the intensity of the coherent light to be having a spatial coherence and the only using the temperol coherence."
#The process is extracted by using the simpy tool

#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation

# Parameters
#wavelength = 0.6328e-6  # Wavelength of light (in meters)
#diameter = 0.01  # Diameter of the aperture (in meters)
#distance = 0.1  # Distance to the observation plane (in meters)
#pixels = 256  # Number of pixels in the observation plane
#size = 0.01  # Size of the observation plane (in meters)
#num_frames = 50  # Number of frames in the time series

# Create a figure for the animation
#fig = plt.figure()


# Function to update the animation frames
#def update(frame):
  #Generate a new random phase pattern for each frame
  #random_phase = np.random.uniform(0, 2 * np.pi, size=(pixels, pixels))

    # Calculate the complex field in the observation plane
    #field = np.exp(1j * (k_x + k_y)) * np.exp(1j * random_phase)

    # Calculate the intensity (speckle pattern)
   #intensity = np.abs(field) ** 2

    # Clear the previous frame and display the new intensity pattern
    #plt.clf()
    #plt.imshow(intensity, cmap='gray')
    #plt.title(f'Frame {frame}')
    #plt.colorbar()


# Generate coordinates in the observation plane
#x = np.linspace(-size / 2, size / 2, pixels)
#y = np.linspace(-size / 2, size / 2, pixels)
#X, Y = np.meshgrid(x, y)

# Calculate the wavevector in the observation plane
#k_x = 2 * np.pi / wavelength * X / distance
#k_y = 2 * np.pi / wavelength * Y / distance

# Create an animation
#ani = animation.FuncAnimation(fig, update, frames=50, interval=1000)

# Display the animation
#plt.show()



#Markov chain, Master Equation, Markov's equation, Numerical methods to find the markov chain.....

#Time history of the speckle pattern (THSP)

#Coocurrance matrix
#Model of the Brownian motion
#recursive function





#differential stochastic process

#Looking at the process of the Fokker planck equation
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Parameters
# D = 1.0  # Diffusion coefficient
# T = 1.0  # Total simulation time
# N = 1000  # Number of time steps
# dt = T / N  # Time step size
# sqrt_dt = np.sqrt(dt)
#
# # Initial condition
# x0 = 0.0
#
# # Arrays to store results
# t_values = np.linspace(0, T, N + 1)
# x_values = np.zeros(N + 1)
#
#
# # Milstein scheme simulation
# x = x0
# x2 = x0
# def good(D,dt, x2):
#     x_values_2 = np.zeros(N + 1)
#     for i in range(N):
#         dW = np.random.normal(0, sqrt_dt)  # Wiener increment
#         drift = -D * x2  # Drift term (Fokker-Planck equation)
#         diffusion = 0.5 * D * D  # Diffusion term (Fokker-Planck equation)
#
#         x2 += drift * dt + diffusion * dW + 0.5 * diffusion * (dW ** 2 - dt)  # Milstein scheme update
#         x_values_2[i + 1] = x2
#     return x_values_2
#
# for i in range(100):
#     good(D, dt, x2)
#
# #
# # for i in range(N):
# #     dW = np.random.normal(0, sqrt_dt)  # Wiener increment
# #     drift = -D * x  # Drift term (Fokker-Planck equation)
# #     diffusion = 0.5 * D * D  # Diffusion term (Fokker-Planck equation)
# #
# #     x += drift * dt + diffusion * dW + 0.5 * diffusion * (dW ** 2 - dt)  # Milstein scheme update
# #     x_values[i + 1] = x
#
# # Plot the simulated speckle process in a single graph
# # plt.plot(t_values, x_values, label='x_values')
# plt.plot(t_values,, label='x_values_2')
# plt.xlabel('Time')
# plt.ylabel('Position')
# plt.title('Simulated Speckle Process')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

#
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from scipy.stats import rice
#
# # Parameters
# image_size = (8, 8)  # Grid size
# #exposure_time = 0.0000015 # Exposure time in seconds
# num_frames = 500 # Number of frames in the movie
#
#
# # Function to simulate speckle intensity for a single frame
# def simulate_frame():
#     sigma = np.random.random()
#     intensity_frame =  np.random.normal(0, sigma, size=image_size)
#     return intensity_frame
#
#
# # Create a figure and axis for the animation
# fig, ax = plt.subplots()
#
# # Initialize the speckle pattern
# intensity_pattern = np.zeros(image_size)
# print(intensity_pattern)
#
#
# # Initialize the image plot
# img = ax.imshow(intensity_pattern, cmap='gray', vmin=-15, vmax=15)
#
# # Function to update the animation frame
# def update(frame):
#     global intensity_pattern
#     intensity_frame = simulate_frame()
#     intensity_pattern += intensity_frame
#     img.set_array(intensity_pattern)
#
# # Create the animation
# ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=5000, blit=False)
#
# # Display the animation (this may take a moment to generate)
#
# plt.show()
#
# np.random.random
#
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define simulation parameters
# grid_size = 100  # Size of the grid (pixels)
# tau_c = 1e-3    # Characteristic decorrelation time (seconds)
# flow_region = (8, 8)  # Region with high flow (pixels)
# high_flow_tau_c = 1e-5  # Decorrelation time in high flow region (seconds)
# exposure_time = 1e-4    # Exposure time (seconds)
#
# # Create a grid
# grid = np.zeros((grid_size, grid_size))
#
# # Add high flow region
# grid[flow_region[0]:flow_region[1], :] = 1
#
# # Simulate speckle intensity
# speckle_intensity = np.exp(-exposure_time / tau_c) * np.random.normal(size=(grid_size, grid_size))
#
# # Display the simulated speckle pattern
# plt.figure(figsize=(6, 6))
# plt.imshow(speckle_intensity, cmap='gray', extent=[0, grid_size, 0, grid_size])
# plt.title('Simulated Laser Speckle Pattern')
# plt.colorbar()
# plt.show()
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Parameters
# image_size = (100, 100)  # Grid size
# μa = 0.1  # Absorption coefficient
# μs_prime = 0.12  # Reduced scattering coefficient
# DB = 1e-8  # Particle diffusion coefficient
# exposure_time = 1e-3  # Exposure time in seconds
#
# # Simulation grid
# intensity = np.ones(image_size)  # Initialize intensity grid with ones
#
# # Simulate speckle intensity using a simplified model
# for _ in range(int(exposure_time / 1e-6)):  # Simulate for 1 ms (adjust as needed)
#     # Generate intensity fluctuations based on diffusion model
#     intensity += np.random.normal(scale=np.sqrt(3 * μa * μs_prime), size=image_size)
#
# # Apply a linear fit (slope should be approximately √3 * μa * μs_prime)
# x = np.arange(image_size[0])
# y = np.mean(np.log(intensity), axis=0)
# fit = np.polyfit(x, y, 1)
# linear_fit = np.polyval(fit, x)
#
# # Plot the results
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 2, 1)
# plt.imshow(np.log(intensity), cmap='gray', extent=[0, 100, 0, 100])
# plt.title("Log Intensity")
# plt.colorbar()
#
# plt.subplot(2, 2, 2)
# plt.plot(x, y, 'b', label='Intensity Profile')
# plt.plot(x, linear_fit, 'r--', label='Linear Fit')
# plt.title("Intensity Profile and Linear Fit")
# plt.xlabel("SD Separation")
# plt.ylabel("Log(φ.r)")
# plt.legend()
#
# # Simulate autocorrelation (for demonstration)
# autocorrelation = np.correlate(intensity.ravel(), intensity.ravel(), mode='full')
#
# plt.subplot(2, 2, 3)
# plt.plot(autocorrelation)
# plt.title("Autocorrelation (Simplified)")
# plt.xlabel("Delay Time")
# plt.ylabel("Autocorrelation")
#
# plt.tight_layout()
# plt.show()
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# # Parameters
# image_size = (6, 6)  # Grid size
# exposure_time = 0.0015   # Exposure time in seconds
# num_frames = 100  # Number of frames in the movie
#
# # Function to simulate speckle intensity for a single frame
# def simulate_frame():
#     intensity_frame = np.random.normal(size=image_size)
#     return intensity_frame
#
# # Create a figure and axis for the animation
# fig, ax = plt.subplots()
#
# # Initialize the speckle pattern
# intensity_pattern = np.ones(image_size)
#
# # Initialize the image plot
# img = ax.imshow(intensity_pattern, cmap='grey', vmin=-15, vmax=15)
#
# # Function to update the animation frame
# def update(frame):
#     global intensity_pattern
#     intensity_frame = simulate_frame()
#     intensity_pattern += intensity_frame
#     img.set_array(intensity_pattern)
#
# # Create the animation
# ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)
#
# # Display the animation (this may take a moment to generate)
# plt.show()




















import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import rice

# Parameters
D = 1.0  # Diffusion coefficient
T = 100  # Total simulation time
N = 1000  # Number of time steps
dt = T / N  # Time step size
sqrt_dt = np.sqrt(dt)
x0 = 0.0

# Arrays to store results
t_values = np.linspace(0, T, N + 1)
x_values = np.zeros(N + 1)

# Milstein scheme simulation
x = x0



for i in range(N):
    dW = np.random.normal(0, sqrt_dt)  # Wiener increment
    drift = -D * x  # Drift term (Fokker-Planck equation)
    diffusion = 0.5 * D * D  # Diffusion term (Fokker-Planck equation)
    x += drift * dt + diffusion * dW + 0.5 * diffusion * (dW ** 2 - dt)  # Milstein scheme update
    x_values[i + 1] = x

# Create a figure and axis for the animation
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Initialize the speckle pattern
image_size = (8, 8)  # Grid size
num_frames = 1500  # Number of frames in the movie

# def simulate_frame():
#     intensity_frame = np.zeros(size = image_size)
#     sigma = np.random.random()
#     intensity_frame_2 = np.random.normal(0, sigma, size=(8,8))
#     return intensity_frame, intensity_frame_2
def simulate_frame():
    sigma = np.random.random()
    intensity_frame = np.random.normal(0, sigma, size=image_size)
    return intensity_frame

intensity_pattern = np.ones(image_size)
# P = (1,1)
# intensity_pattern[1,1]= 130


# Initialize the image plot for the speckle pattern
img1 = ax[0].imshow(intensity_pattern, cmap='gray', vmin=-15, vmax=15)


# Function to update the speckle pattern animation frame



def update_speckle(frame):
    global intensity_pattern
    intensity_frame = simulate_frame()
    intensity_pattern += intensity_frame
    img1.set_array(intensity_pattern)

# def update_speckle(frame):
#     global intensity_pattern
#     intensity_frame = simulate_frame()
#     intensity_pattern += intensity_frame
#     img1.set_array(intensity_pattern)

# Create the speckle pattern animation
ani1 = animation.FuncAnimation(fig, update_speckle, frames=num_frames, interval= 1000, blit=False)

# Plot the simulated speckle process
ax[1].plot(t_values, x_values)
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Position')
ax[1].set_title('Simulated Speckle Process')
ax[1].grid(True)

# Display both animations (this may take a moment to generate)
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Parameters
D = 1.0  # Diffusion coefficient
T = 1.0  # Total simulation time
N = 1000  # Number of time steps
dt = T / N  # Time step size
sqrt_dt = np.sqrt(dt)

# Initial condition
x0 = 0.0

# Arrays to store results
t_values = np.linspace(0, T, N + 1)
x_values = np.zeros(N + 1)

# Milstein scheme simulation
x = x0

for i in range(N):
    dW = np.random.normal(0, sqrt_dt)  # Wiener increment
    drift = -D * x  # Drift term (Fokker-Planck equation)
    diffusion = 0.5 * D * D  # Diffusion term (Fokker-Planck equation)

    x += drift * dt + diffusion * dW + 0.5 * diffusion * (dW ** 2 - dt)  # Milstein scheme update
    x_values[i + 1] = x

# Plot the simulated speckle process
plt.plot(t_values, x_values, label='x_values')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Simulated Speckle Process')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
# (your previous parameters)

# Pixel properties
pixel_position = [4, 4]  # Initial position of the pixel (row, column)
pixel_motion = [1, 0]  # Motion of the pixel in each frame (row, column)


# Arrays to store results
# (your previous arrays)

# Function to update the speckle pattern animation frame
def update_speckle(frame):
    global intensity_pattern, pixel_position
    intensity_frame = simulate_frame()
    intensity_pattern += intensity_frame

    # Update the pixel's position based on its motion
    pixel_position[0] += pixel_motion[0]
    pixel_position[1] += pixel_motion[1]

    # Ensure the pixel remains within the image bounds
    pixel_position[0] = np.clip(pixel_position[0], 0, image_size[0] - 1)
    pixel_position[1] = np.clip(pixel_position[1], 0, image_size[1] - 1)

    # Update the pixel's position in the intensity pattern
    intensity_pattern[int(pixel_position[0]), int(pixel_position[1])] += 1.0

    img1.set_array(intensity_pattern)


# Create the speckle pattern animation
ani1 = animation.FuncAnimation(fig, update_speckle, frames=num_frames, interval= 1000, blit=False)

# Plot the simulated speckle process
# (your previous plot code)

# Display both animations
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
D = 1000 # Diffusion coefficient
T = 1.0  # Total simulation time
N = 1000  # Number of time steps
dt = T / N  # Time step size
sqrt_dt = np.sqrt(dt)

# Image properties
image_size = (8, 8)  # Size of the 2D image
num_pixels = image_size[0] * image_size[1]  # Total number of pixels

# Initial conditions for particle positions
num_particles = 1
initial_positions = np.random.uniform(0, 10, size=(num_particles, 2))
particle_positions = initial_positions.copy()

# Arrays to store results
t_values = np.linspace(0, T, N + 1)
intensity_pattern = np.zeros(image_size)  # Speckle pattern image

# Milstein scheme simulation
for i in range(N):
    dW = np.random.normal(0, sqrt_dt, size=(num_particles, 1))  # Wiener increments for each particle
    drift = -D * particle_positions  # Drift term (Fokker-Planck equation)
    diffusion = 0.5 * D * D  # Diffusion term (Fokker-Planck equation)

    particle_positions += drift * dt + diffusion * dW + 0.5 * diffusion * (dW ** 2 - dt)  # Milstein scheme update

    # Update the intensity pattern based on particle positions
    for j in range(num_particles):
        x, y = particle_positions[j].astype(int)
        x = np.clip(x, 0, image_size[0] - 1 )
        y = np.clip(y, 0, image_size[1] - 1)
        intensity_pattern[x, y] += 1

    # Plot the intensity pattern for each time step
    plt.imshow(intensity_pattern, cmap='gray', extent=[0, image_size[1], 0, image_size[0]], origin='lower')
    plt.title(f'Time Step {i + 1}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.pause(0.1)
    plt.clf()

# Show the final intensity pattern
plt.imshow(intensity_pattern, cmap='gray', extent=[0, image_size[1], 0, image_size[0]], origin='lower')
plt.title('Final Speckle Pattern')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()
