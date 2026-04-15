require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const multer = require("multer");
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

// MongoDB connection
mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  serverSelectionTimeoutMS: 30000,
  socketTimeoutMS: 45000,
  connectTimeoutMS: 45000,
  family: 4,
  retryWrites: true,
})
  .then(() => console.log("MongoDB Connected"))
  .catch((error) => console.error("Connection error:", error));

// Mongoose schema and model for storing images and QR data
const imageSchema = new mongoose.Schema({
  qrData: {
    rollNo: String,
    name: String,
    fatherName: String,
    department: String,
    contact: String,
  },
  images: [
    {
      img: Buffer,
      imgType: String,
    },
  ],
});

const ImageDocument = mongoose.model("ImageDocument", imageSchema);

// Set up multer for file handling in memory
const storage = multer.memoryStorage();
const upload = multer({ storage });

// Endpoint for uploading images
app.post("/api/uploadImages", upload.array("images", 10), async (req, res) => {
  const qrData = JSON.parse(req.body.qrData);  // Parse the structured QR data

  const rollNo = qrData.rollNo;

  // Define the folder where images will be saved, using roll number as subfolder
  const uploadDir = path.join(__dirname, "dataset", rollNo);

  // Create the directory if it doesn't exist
  if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
  }

  try {
    // Prepare the images array and save them to the disk
    const imagesArray = await Promise.all(
      req.files.map(async (file) => {
        const filePath = path.join(uploadDir, file.originalname);

        // Save the file to disk
        fs.writeFileSync(filePath, file.buffer);

        return {
          img: file.buffer,
          imgType: file.mimetype,
        };
      })
    );

    // Find an existing document with the same rollNo or create a new one
    let imageDoc = await ImageDocument.findOne({ "qrData.rollNo": qrData.rollNo });

    if (imageDoc) {
      // Update existing document by appending new images
      imageDoc.images.push(...imagesArray);
    } else {
      // Create a new document with structured qrData and images array
      imageDoc = new ImageDocument({
        qrData,
        images: imagesArray,
      });
    }

    // Save the document to MongoDB
    await imageDoc.save();

    // Send a success response
    res.status(201).json({
      message: "Images uploaded successfully",
      qrData,
      files: imagesArray.length,
    });
  } catch (error) {
    console.error("Error saving to MongoDB or file system:", error);
    res.status(500).json({ error: "Failed to upload images" });
  }
});// Endpoint to train a specific user's dataset


app.post('/api/trainUser', async (req, res) => {
  const { rollNo } = req.body;

  if (!rollNo) {
    return res.status(400).json({ error: 'Roll number is required' });
  }

  const datasetPath = path.join(__dirname, 'dataset', rollNo);

  if (!fs.existsSync(datasetPath)) {
    return res.status(404).json({ error: `No dataset found for roll number ${rollNo}` });
  }

  // Call the Python script with the specific rollNo dataset path
  const pythonProcess = spawn('python', [path.join(__dirname, 'generate_encodings.py'), datasetPath]);

  pythonProcess.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    if (code === 0) {
      res.status(200).send({ message: 'User-specific model trained successfully' });
    } else {
      res.status(500).send({ error: 'Failed to train model for user' });
    }
  });
});

const port = process.env.PORT || 4000;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});








// const express = require("express");
// const mongoose = require("mongoose");
// const bodyParser = require("body-parser");
// const cors = require("cors");
// require("dotenv").config();

// const app = express();
// const PORT = process.env.PORT || 4000;

// // Middleware
// app.use(cors());
// app.use(bodyParser.json());

// // MongoDB Connection
// mongoose
//   .connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
//   .then(() => console.log("Connected to MongoDB"))
//   .catch((err) => console.error("Error connecting to MongoDB:", err));

// // MongoDB Schema and Model
// const UserDetailsSchema = new mongoose.Schema({
//   rollNo: { type: String, required: true },
//   name: { type: String, required: true },
//   fatherName: { type: String, required: true },
//   department: { type: String, required: true },
//   contact: { type: String, required: true },
// });


// const UserDetails = mongoose.model("UserDetails", UserDetailsSchema);

// // API Endpoint to Save QR Data
// app.post("/api/saveDetails", async (req, res) => {
//   const { rollNo, name, fatherName, department, contact } = req.body;

//   if (!rollNo || !name || !fatherName || !department || !contact) {
//     return res.status(400).json({ message: "All fields are required" });
//   }

//   try {
//     // Check if user already exists
//     const existingUser = await UserDetails.findOne({ rollNo });
//     if (existingUser) {
//       return res.status(400).json({ message: "User with this roll number already exists" });
//     }

//     // Save new user
//     const newUser = new UserDetails({ rollNo, name, fatherName, department, contact });
//     await newUser.save();

//     res.status(201).json({ message: "Details saved successfully", user: newUser });
//   } catch (error) {
//     console.error("Error saving details:", error);
//     res.status(500).json({ message: "Internal Server Error" });
//   }
// });

// // Start the Server
// app.listen(PORT, () => {
//   console.log(`Server running on http://localhost:${PORT}`);
// });
