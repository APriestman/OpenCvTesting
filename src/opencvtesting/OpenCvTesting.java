/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package opencvtesting;

import java.io.File;
import java.util.logging.Level;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;

/**
 *
 * @author apriestman
 */
public class OpenCvTesting {

    private static final String MODEL = "R:\\work\\images\\Object Detection\\bvlc_googlenet.caffemodel";
    private static final String PROTOTXT = "R:\\work\\images\\Object Detection\\bvlc_googlenet.prototxt";
    private static final String LABELS = "R:\\work\\images\\Object Detection\\synset_words.txt";
    //private static final String INPUT = "R:\\work\\images\\Object Detection\\cat1.jpg";
    //private static final String INPUT = "R:\\work\\images\\Object Detection\\space_shuttle.jpg";
    //private static final String INPUT = "R:\\work\\images\\Object Detection\\person.jpg";
    private static final String INPUT = "R:\\work\\images\\Object Detection\\bird2.jpg";
    //private static final String INPUT = "R:\\work\\images\\Object Detection\\a.png";
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        if (!OpenCvLoader.isOpenCvLoaded()) {
            System.out.println("OpenCV is not loaded");
            return;
        }
        
        System.out.println("\n#######\nModel testing");
        File modelFile = new File(MODEL);
        if (modelFile.exists()) {
            System.out.println("Found model file " + modelFile.getAbsolutePath());
        } else {
            System.out.println("Could not find model file " + modelFile.getAbsolutePath());
            return;
        }
        File protoFile = new File(PROTOTXT);
        if (protoFile.exists()) {
            System.out.println("Found prototxt file " + protoFile.getAbsolutePath());
        } else {
            System.out.println("Could not find prototxt file " + protoFile.getAbsolutePath());
            return;
        }
        File labelFile = new File(LABELS);
        if (labelFile.exists()) {
            System.out.println("Found labels file " + labelFile.getAbsolutePath());
        } else {
            System.out.println("Could not find labels file " + labelFile.getAbsolutePath());
            return;
        }
        File inputImage = new File(INPUT);
        if (inputImage.exists()) {
            System.out.println("Found image file " + inputImage.getAbsolutePath());
        } else {
            System.out.println("Could not find image file " + inputImage.getAbsolutePath());
            return;
        }
        
        List<String> classNames = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(LABELS))) {
                String line = reader.readLine();
                while (line != null) {
                        line = line.substring(line.indexOf(" "));
                        classNames.add(line);
                        // read next line
                        line = reader.readLine();
                }
                reader.close();
        } catch (IOException e) {
                e.printStackTrace();
        }
        
        try {
            System.out.println("Loading model");
            Net net = Dnn.readNetFromCaffe(protoFile.getAbsolutePath(), modelFile.getAbsolutePath());
            net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
            net.setPreferableTarget(Dnn.DNN_TARGET_CPU);
            if (net.empty()) {
                System.out.println("Net is empty");
                return;
            }
            
            Mat originalImage = Imgcodecs.imread(INPUT, Imgcodecs.IMREAD_COLOR);
            Mat convImage = new Mat();
            originalImage.convertTo(convImage, CvType.CV_32FC3);
            
            //Mat blob = Dnn.blobFromImage(convImage);
            Mat blob = Dnn.blobFromImage(convImage, 
                    1, new Size(224,224), new Scalar(0, 0, 0), false, false);
            //Mat blob = Dnn.blobFromImage(convImage, 
            //        1.0/255, new Size(224,224), new Scalar(104, 117, 123), false, false); 
            
            System.out.println("Running model");
            net.setInput(blob);
            //net.setInput(blob, "data"); // I believe this has to match the "input" name in the prototxt
            //Mat result = net.forward();            
            Mat result = net.forward("prob");
            //Mat result = net.forward("loss3/classifier"); // Also from the prototxt file
            
            //System.out.println("result rows: " + result.rows() + ", cols: " + result.cols());
            double maxProb = 0;
            int maxIndex = 0;
            for (int i = 0;i < result.cols();i++) {
                Mat col = result.col(i);
                //System.out.println("Col mat dump: " + col.dump());
                //System.out.println("Col mat [0]: " + col.get(0, 0)[0]);
                double prob = col.get(0, 0)[0];
                if (prob > maxProb) {
                    maxProb = prob;
                    maxIndex = i;
                    //System.out.println("Setting max to column " + maxIndex + " with prob " + maxProb);
                }
            }
            System.out.println("Max column " + maxIndex + " with prob " + maxProb);
            System.out.println("Best match: " + classNames.get(maxIndex));
            
            Mat probMat = result.reshape(1, 1);
            Core.MinMaxLocResult minMaxRes = Core.minMaxLoc(probMat);
            //System.out.println("minVal: " + minMaxRes.minVal + "   maxVal: " + minMaxRes.maxVal);
            //System.out.println("minLoc: " + minMaxRes.minLoc + "   maxLoc: " + minMaxRes.maxLoc);
                        
            
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        
    }
    
}
