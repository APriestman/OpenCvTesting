/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package opencvtesting;

import org.opencv.core.Core;

public final class OpenCvLoader {

    private static final boolean OPEN_CV_LOADED;
    private static UnsatisfiedLinkError exception = null;

    static {
        boolean tempOpenCvLoaded = false;
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            tempOpenCvLoaded = true;
        } catch (UnsatisfiedLinkError e) {
            tempOpenCvLoaded = false;
            exception = e;  //save relevant error for throwing at appropriate time
        }
        OPEN_CV_LOADED = tempOpenCvLoaded;
    }

    /**
     * Return whether or not the OpenCV library has been loaded.
     *
     * @return - true if the opencv library is loaded or false if it is not
     */
    public static boolean isOpenCvLoaded() throws UnsatisfiedLinkError {
        if (!OPEN_CV_LOADED) {
             //exception should never be null if the open cv isn't loaded but just in case
            if (exception != null) {
                throw exception;
            } else {
                throw new UnsatisfiedLinkError("OpenCV native library failed to load");
            }

        }
        return OPEN_CV_LOADED;
    }
}
