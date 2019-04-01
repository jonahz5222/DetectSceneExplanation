# Detect Scene Explanation

## Introduction
The given [app](https://content.techinnovator.info/mu/sp19/INFOTC4445/Machine%20Learning%20Vision/ML-Vision.zip) utilizes the CoreML framework provided by Apple to categorize images and determine the contents of said photos.

## detectScene(image: CIImage)
The first line of detectScene (`displayString(string: "detecting scene...")`) simply calls another function that writes text to a TextView. This serves the purpose of notifying the user that the image classifier has begun running.
  
This next code block creates the model that we will be using to classify the image. On error it informs the user that it has done such utilizing the aforementioned function.
```swift
guard let model = try? VNCoreMLModel(for: VGG16().model) else {
    displayString(string: "Can't load ML model.")
    return
}
```
This guard let calls the failable initializer for a CoreML Model (`VNCoreMLModel(for: VGG16.model)`. Specifically one to be used with Vision requests (Vision is another framework provided by Apple underneath the CoreML umbrella). It is passed the model that is included with the project in the Resources group.
  
It may seem weird that this .mlmodel file is able to be accessed like it is a class in our code. This is because Swift automatically generates a class for our model when it is added. This gives us access to that `.model` property. 

Next, we create a request. This request, by utilizing the model we created, is what actually processes our image.

```swift
let request = VNCoreMLRequest(model: model) { [weak self] request, error in
    guard let results = request.results as? [VNClassificationObservation],
        let _ = results.first else {
            self?.displayString(string: "Unexpected result type from VNCoreMLRequest")
            return
    }

    // Update UI on main queue
    DispatchQueue.main.async { [weak self] in
        self?.activityIndicator.stopAnimating()
        for result in results {
            self?.displayString(string: "\(Int(result.confidence * 100))% \(result.identifier)")
        }
    }
}
```

The bulk of this code block is not the actual creation of a request, but rather the completion handler that runs immediately after the request is finished. Lets talk about that completion handler.
  
This is the first line of the completion handler:
```swift
[weak self] request, error in
```
request and error are parameters to be passed to the handler, and their purpose is indicated by their naming. `[weak self]` however, is kind of weird. This is a precaution taken when using completion handlers to avoid the creation of something called a 'retain cycle' or in more common language, a memory leak. A more detailed explanation can be found [here](https://benscheirman.com/2018/09/capturing-self-with-swift-4-2/). This ensures that unless we strongly reference `self` inside of the completion block, it will be nil. By labeling self as `weak`, we're essentially telling the completion block to make this reference (the one in the block) nil if the original self (the one outside of the block) is released by the memory management system.
  
Once inside of the completion handler, we attempt to access the results provided by the now finished request.
```swift
guard let results = request.results as? [VNClassificationObservation],
    let _ = results.first else {
        self?.displayString(string: "Unexpected result type from VNCoreMLRequest")
        return
}
```
First, by accessing the results property on the request object (passed into the completion handler as a parameter), we attempt to cast the results as `[VNClassificationObservation]`. This ensures the the results fit our desired format. On the next line of the guard let, we ensure that `results` contains values by checking if the first character exists. If either of these checks fail, we display an error to the user and exit.
  
Next, we update the label with our new results:
```swift
DispatchQueue.main.async { [weak self] in
    self?.activityIndicator.stopAnimating()
    for result in results {
        self?.displayString(string: "\(Int(result.confidence * 100))% \(result.identifier)")
    }
}
```
This is done on the main thread, as is required when we are updating any part of the UI. It is necessary for us to use `DispatchQueue.main.async` to access the main thread as we are currently in a completion handler. We stop the activity indicator (you might be wondering why we are stopping the activity indicator before we start it seemingly. This is because the completion handler runs asynchronously and by the time this code actually runs, we *will* have started the activity indicator). We iterate through the results and add each one to our UI, along with the confidence property provided by the result.
  
Next, we actually perform the request: 
```swift
activityIndicator.startAnimating()

// Run the Core ML GoogLeNetPlaces classifier on global dispatch queue
let handler = VNImageRequestHandler(ciImage: image)
DispatchQueue.global(qos: .userInteractive).async {
    do {
        try handler.perform([request])
    } catch {
        DispatchQueue.main.async { [weak self] in
            self?.displayString(string: error.localizedDescription)
            self?.activityIndicator.stopAnimating()
        }
    }
}
```
Naturally, we start the activity indicator, as this process willl take a bit. Then, we create the handler. This is the object that will actually perform the request that we created above. It is passed the image that it will perform the request upon. Next, while running on the `DispatchQueue.global`, which ensures that our code will run concurrently, we create a do catch block, as performing the request could throw an error. Inside of the do catch, we perform the request: `try handler.perform([request])`. `request` is within brackets because the perform function takes an array of VNRequests, so we must pass it as an array (even if it is the only object in it). If this throws an error, or fails, we update the UI on the main thread with the error's description and stop the activity indicator. If it does not fail, the request completion handler is called. detectScene has completed. 
