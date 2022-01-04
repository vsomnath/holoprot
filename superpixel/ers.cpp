#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "MERCLazyGreedy.h"
#include "MERCInputGraph.h"
#include "MERCOutput.h"

namespace py = pybind11;

using namespace std;

class ERS {
public:
    ERS( double lam) { lambda_ = lam;}
    double lambda_;

    py::array_t<float> computeSegmentation(py::array_t<int> edgeList, py::array_t<float> edgeSim, int numNodes, int nC) {
        MERCInputGraph<int> input;
        int kernel = 0;

        // read the graph for segmentation
        input.ReadGraph(edgeList, edgeSim, numNodes);

        // entropy rate superpixel segmentation
        MERCLazyGreedy merc;
        merc.ClusteringTreeIF(input.nNodes_, input, kernel, lambda_*1.0*nC, nC);
        vector<int> label = MERCOutput::DisjointSetToLabel(merc.disjointSet_);

        py::array_t<float> result = py::array_t<float>(input.nNodes_);
        py::buffer_info buf_out = result.request();
        float *out = (float *) buf_out.ptr;

        for (int num = 0; num < input.nNodes_; num++){
          out[num] = (float)label[num];
        }
        return result;
    }
};


PYBIND11_MODULE(ers, m) {
    m.doc() = "Python Bindings for Entropy Rate Superpixel";

    py::class_<ERS>(m, "ERS")
        .def(py::init<double>())
        .def("computeSegmentation", &ERS::computeSegmentation);
}
