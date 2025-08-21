sample=cv2.imread('your_img.png', 0)     #insert file path here
sample=cv2.resize(sample, (28, 28))
sample=sample/255
plt.imshow(sample, cmap='gray')

sample=np.reshape(sample, (1, 28, 28))

s_pred=model(sample, training=False)
print("The number you wrote is: ", np.argmax(s_pred))
