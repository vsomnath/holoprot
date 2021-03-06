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
#ifndef _m_erclustering_h_
#define _m_erclustering_h_

#include "MERCDisjointSet.h"
#include "MERCInput.h"
#include "MERCEdge.h"
#include "MSubmodularHeap.h"
#include "MERCFunctions.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;


class MERCClustering
{
public:

	void ClusteringTreeIF(int nVertices,MERCInput &edges,int kernel,double lambda,int nC)
	{
		disjointSet_ = ClusteringTree(nVertices,edges,kernel,lambda,nC);
	};

	virtual MERCDisjointSet* ClusteringTree(int nVertices,MERCInput &edges,int kernel,double lambda,int nC) = 0;

	MERCClustering()
	{
		disjointSet_ = NULL;
	};

	~MERCClustering()
	{
		Release();
	};

	void Release()
	{
		if(disjointSet_)
			delete disjointSet_;
	};

	MERCDisjointSet *disjointSet_;

};

#endif
