/*
Copyright 2011, Ming-Yu Liu

All Rights Reserved

Permission to use, copy, modify, and distribute this software and
its documentation for any non-commercial purpose is hereby granted
without fee, provided that the above copyright notice appear in
all copies and that both that copyright notice and this permission
notice appear in supporting documentation, and that the name of
the author not be used in advertising or publicity pertaining to
distribution of the software without specific, written prior
permission.

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
ANY PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/
#ifndef _m_erc_input_graph_h_
#define _m_erc_input_graph_h_

#include "MERCInput.h"
#include <cmath>

namespace py = pybind11;

using namespace std;

template <class T>
class MERCInputGraph: public MERCInput
{
public:
	void ReadGraph(py::array_t<int> edgeList, py::array_t<double> edgeSim, int numNodes);
};

template <class T>
void MERCInputGraph<T>::ReadGraph(py::array_t<int> edgeList, py::array_t<double> edgeSim, int numNodes){
		auto bufList = edgeList.unchecked<2>();
		py::buffer_info bufSim = edgeSim.request();
		double *bufSimPtr = (double *) bufSim.ptr;

		nEdges_ = bufList.shape(0);
		nNodes_ = numNodes;
		edges_ = new Edge [nEdges_];

		for (int num = 0; num < nEdges_; num++){
			edges_[num].a_ = bufList(num, 0);
			edges_[num].b_ = bufList(num, 1);
			edges_[num].w_ = bufSimPtr[num];
		}

}

#endif
