### 8/29/24 A visualization utils file, long overdue
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns


def display_ind_barcode(vector,cmap='gray',figsize=10):
    # Normalize vector values to range [0, 1]
    #normalized_vector = (vector - vector.min()) / (vector.max() - vector.min())
    normalized_vector = vector
    #print(normalized_vector.size)
    
    # Set the height of the barcode
    barcode_height = 10 

    # Create an empty canvas for the barcode
    barcode = np.zeros((barcode_height, 1 * len(vector)))

    # Fill the barcode with rectangles corresponding to the vector values
    for i, val in enumerate(normalized_vector):
        barcode[:, i:(i+1)] = val

    # Create subplots for displaying image and barcode
    fig = plt.figure(figsize=figsize)
    axs = plt.subplots()

    # Display the barcode
    axs.imshow(barcode, cmap=cmap)
    axs.axis('off')

    plt.show()


def display_image_and_barcode(pixel_values, vector):
    # Ensure pixel_values and vector are numpy arrays
    pixel_values = np.array(pixel_values)
    image = np.moveaxis(pixel_values,0,-1)
    vector = np.array(vector)
    

    # Check if the shape of pixel_values is correct
    if len(pixel_values.shape) != 3:
        raise ValueError("Input tensor shape must be (channels, height, width)")

    # #Transpose the tensor to the correct shape for displaying with matplotlib
    # image = np.transpose(pixel_values, (1, 2, 0))

    # Normalize vector values to range [0, 1]
    normalized_vector = (vector - vector.min()) / (vector.max() - vector.min())
   
    # Set the height of the barcode
    barcode_height = 20

    # Create an empty canvas for the barcode
    barcode = np.zeros((barcode_height, 1 * len(vector)))

    # Fill the barcode with rectangles corresponding to the vector values
    for i, val in enumerate(normalized_vector):
        barcode[:, i:(i+1)] = val

    # Create subplots for displaying image and barcode
    fig, axs = plt.subplots(2,1)

    # Display the image
    axs[0].imshow(image)
    axs[0].axis('off')

    # Display the barcode
    axs[1].imshow(barcode, cmap='gray')
    axs[1].axis('off')

    plt.show()

def display_w_layer(w,cmap='gray'):
    #display weights of model layer

    if len(w.shape) <= 2:
        height, width = w.shape
        print("Weight layer has shape",w.shape)
    else:
        print("Weight layer is not two dimensional")
        return
    
    w = (2* (w - w.min()) / (w.max() - w.min()) )- 1
    
    # Create subplots for displaying 
    fig, axs = plt.subplots(figsize=(10,20))

    # Display the weight layer
    axs.imshow(w.T, cmap=cmap)
    axs.axis('off')

    plt.show()

def display_histogram(w, bins=50, color='blue'):
    """
    Display a histogram of the values in the weight layer.
    
    Parameters:
    - w: 2D array-like object containing the weights.
    - bins: Number of bins for the histogram.
    - color: Color of the histogram bars.
    """
    
    if len(w.shape) <= 2:
        height, width = w.shape
        print("Weight layer has shape", w.shape)
    else:
        print("Weight layer is not two-dimensional")
        return
    
    # Flatten the 2D array to 1D for histogram plotting
    w_flat = w.flatten()
    
    # Plot the histogram
    plt.figure()
    plt.hist(w_flat, bins=bins, color=color, alpha=0.75)
    
    # Add labels and title
    plt.xlabel('Weight values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Weight Layer Values')
    
    # Show the plot
    plt.show()

def display_rgba_histogram(w_rgba, bins=50, color='blue'):
    """
    Display a histogram of the values in the weight layer.
    
    Parameters:
    - w: 2D array-like object containing the weights.
    - bins: Number of bins for the histogram.
    - color: Color of the histogram bars.
    """
    
    if len(w.shape) <= 2:
        height, width = w.shape
        print("Weight layer has shape", w.shape)
    else:
        print("Weight layer is not two-dimensional")
        return
    
    # Flatten the 2D array to 1D for histogram plotting
    w_flat = w.flatten()
    
    # Plot the histogram
    plt.figure()
    plt.hist(w_flat, bins=bins, color=color, alpha=0.75)
    
    # Add labels and title
    plt.xlabel('Weight values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Weight Layer Values')
    
    # Show the plot
    plt.show()

def show_conf_matrix(cm,truth_labels,pred_labels):
    '''
    Plot a confusion matrix of a sklearn.metrics import confusion_matrix object
    
    Parameters: 
    - cm (a confusion matrix)
    '''
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, xticklabels=np.unique(truth_labels), yticklabels=np.unique(pred_labels))

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# def showemb(data,cmap="Blues"):
#   """Display all embeddings in a df as a colormap"""
#   # data = df.iloc[:,3:].values
#   ilocs = []
#   prev = -1
#   ids = df2['ID']
#   for i in range(data.shape[0]):
#     if ids.iloc[i]!=prev: ilocs.append( (i,ids.iloc[i]) )
#     prev = ids.iloc[i]
#   ilocs = np.array(ilocs)
#   ids = ilocs[:,1]
#   ilocs = ilocs[:,0]

#   fig = plt.figure(figsize=(20,10))
#   plt.imshow(data.T,cmap=cmap)
#   gridpos = ilocs #np.arange(0,data.shape[0],3)
#   #ids = df.loc[gridpos.astype(np.uint16),'ID'].values
#   plt.xticks(gridpos-0.5, ids)
#   fr=np.arange(0,128.1,16, dtype=int)
#   plt.yticks(fr-0.5, fr)
#   plt.grid('x',color='r')
#   plt.xlabel('samples for each ID')
#   plt.ylabel('features')
#   plt.axis('auto')
#   plt.tight_layout()

def showemb(df,cmap="Blues"):
  '''
  Display a colormap of embeddings grouped by identity
  Params:
    df (pd.DataFrame): dataframe sorted by ID with columns for ID and features_x

  Displays a plt.figure. 
  '''
  data = df.iloc[:,1:].values
  ilocs = []
  prev = -1
  ids = df['ID']
  for i in range(data.shape[0]):
    if ids.iloc[i]!=prev: 
      ilocs.append((i,ids.iloc[i]))
      print("new id at"+str(i))
    prev = ids.iloc[i]
  ilocs = np.array(ilocs)
  ids = ilocs[:,1]
  ilocs = ilocs[:,0]

  fig = plt.figure(figsize=(20,10))
  plt.imshow(data.T,cmap=cmap)
  gridpos = ilocs #np.arange(0,data.shape[0],3)
  #ids = df.loc[gridpos.astype(np.uint16),'ID'].values
  plt.xticks(gridpos-0.5, ids)
  #plt.yticks(fr-0.5, fr)
  plt.grid('x',color='r')
  plt.xlabel('samples for each ID')
  plt.ylabel('features')
  plt.axis('auto')
  plt.tight_layout()

def getgridinfo(df):
  ilocs = []
  prev = -1
  ids = df['ID']
  for i in range(df.shape[0]):
    if ids.iloc[i]!=prev: ilocs.append( (i,ids.iloc[i]) )
    prev = ids.iloc[i]
  ilocs = np.array(ilocs)
  ticklabels = ilocs[:,1]
  ticks = ilocs[:,0]
  return ticks, ticklabels

def showdistances(Aa, ticks, ticklabels):
  fig = plt.figure(figsize=(20,16))
  plt.imshow(Aa)
  gridpos = ticks
  ids = ticklabels
  plt.xticks(gridpos-0.5, ids)
  plt.yticks(gridpos-0.5, ids)
  plt.grid(color='r')
  plt.title('')
  #plt.clim([0,s])
  plt.colorbar()