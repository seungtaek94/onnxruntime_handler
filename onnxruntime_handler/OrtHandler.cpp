
#include <algorithm>

#include "onnxruntime_cxx_api.h"
#include "OrtHandler.h"
#include "OrtHandlerCore.h"


namespace Ort {
    Handler::Handler(std::string model_path) {
        core = (OrtHandlerCore *) new OrtHandlerCore(model_path);
    }

    Handler::~Handler() {
        delete (OrtHandlerCore *) core;
    }

    std::vector<const char *> Handler::GetInputNames() {
        return ((OrtHandlerCore *) core)->GetInputNames();
    }

    std::vector<const char *> Handler::GetOutputNames() {
        return ((OrtHandlerCore *) core)->GetOutputNames();
    }

    void Handler::blobFromImageData(
            float *data, const int rows, const int cols, const int ch,
            const std::vector<float> mean,
            const std::vector<float> std,
            const bool swapRB)
    {
        const size_t nPixels = rows * cols;

        float *flatten = new float[nPixels * ch];

        int nPos = 0;
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col += 1)
            {
                int nIdx = 0;
                nIdx = row * cols * 3 + col * 3;

                flatten[nPos] = (data[nIdx] - mean[0]) / std[0];                      // Blue
                flatten[nPos + nPixels] = (data[nIdx + 1] - mean[1]) / std[1];        // Grean
                flatten[nPos + (nPixels * 2)] = (data[nIdx + 2] - mean[2]) / std[2];  // Red

                if (swapRB) // Swap B <-> R Channel
                {
                    std::swap(flatten[nPos], flatten[nPos + (nPixels * 2)]);
                }
                nPos++;
            }
        }
        std::copy_n(flatten, (nPixels * ch), data);

        delete[] flatten;
    }
}
