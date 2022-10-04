
#include <algorithm>
#include <memory>

#include "onnxruntime_cxx_api.h"
#include "OrtHandler.h"
#include "OrtHandlerCore.h"


namespace Ort {
    Handler::Handler() {
        core = (OrtHandlerCore *) new OrtHandlerCore();
    }

    Handler::~Handler() {
        delete (OrtHandlerCore *) core;
    }

    std::unique_ptr<Handler> Handler::LoadModel(std::string model_path)
    {
        std::unique_ptr<Handler> handler(new Handler);
        ((OrtHandlerCore *) handler->core)->LoadModel(model_path);

        return std::move(handler);
    }

    std::vector<const char *> Handler::GetInputNames() {
        return ((OrtHandlerCore *) this->core)->GetInputNames();
    }

    std::vector<const char *> Handler::GetOutputNames() {
        return ((OrtHandlerCore *) this->core)->GetOutputNames();
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
