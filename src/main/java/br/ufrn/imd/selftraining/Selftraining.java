package br.ufrn.imd.selftraining;

import java.util.ArrayList;

import br.ufrn.imd.selftraining.bc.Classifier;
import br.ufrn.imd.selftraining.bc.DecisionTable;
import br.ufrn.imd.selftraining.bc.Ibk;
import br.ufrn.imd.selftraining.bc.J48;
import br.ufrn.imd.selftraining.bc.NaiveBayes;
import br.ufrn.imd.selftraining.bc.Smo;
import br.ufrn.imd.selftraining.enums.DecisionTableSParameter;


public class Selftraining {

	public static ArrayList<Classifier> clasifierPool;
	public static int labeledPercentual = 10;
	
	public static void main(String args[]) {
		clasifierPool = new ArrayList<Classifier>();
		buildClassifiersForSelfTraining();
	}
	
	public static void buildClassifiersForSelfTraining() {
		
		J48 j48a = new J48();
		J48 j48b = new J48();
		J48 j48c = new J48();
		J48 j48d = new J48();
		
		NaiveBayes nb1 = new NaiveBayes();
		NaiveBayes nb2 = new NaiveBayes();
		NaiveBayes nb3 = new NaiveBayes();
		
		Ibk ibk1 = new Ibk();
		Ibk ibk2 = new Ibk();
		Ibk ibk3 = new Ibk();
		Ibk ibk4 = new Ibk();
		Ibk ibk5 = new Ibk();
		
		Smo smo1 = new Smo();
		Smo smo2 = new Smo();
		Smo smo3 = new Smo();
		Smo smo4 = new Smo();
		Smo smo5 = new Smo();
		
		DecisionTable dt1 = new DecisionTable();
		DecisionTable dt2 = new DecisionTable();
		DecisionTable dt3 = new DecisionTable();
		
		//OK
		j48a.setParameters(false, false, false, false, false, false, 2, 0.05);
		j48b.setParameters(false, false, false, false, false, false, 2, 0.10);
		j48c.setParameters(false, false, false, false, false, false, 2, 0.20);
		j48d.setParameters(false, false, false, false, false, false, 2, 0.25);
		
		//OK
		nb1.setParameters(false, false);
		nb2.setParameters(true, false);
		nb3.setParameters(false, true);
		/*
		///boolean E, boolean I, boolean F, boolean X, int K, String A
		ibk1.setParameters(false, false, false, false, 1,IbkAParameter.EUCLIDIAN.getInfo());
		ibk2.setParameters(false, false, false, false, 3,IbkAParameter.EUCLIDIAN.getInfo());
		ibk3.setParameters(false, false, false, false, 3,IbkAParameter.MANHATAM.getInfo());
		ibk4.setParameters(false, false, false, false, 5,IbkAParameter.EUCLIDIAN.getInfo());
		ibk5.setParameters(false, false, false, false, 5,IbkAParameter.MANHATAM.getInfo());
		
		smo1.setParameters(SmoSelParameter.POLY_KERNEL.getInfo(),1);
		smo2.setParameters(SmoSelParameter.NORMALIZED_POLY_KERNEL.getInfo(),1);
		smo3.setParameters(SmoSelParameter.RBF_KERNEL.getInfo(),1);
		smo4.setParameters(SmoSelParameter.PUK.getInfo(),1);
		smo5.setParameters(SmoSelParameter.POLY_KERNEL.getInfo(),.8);
		*/
		dt2.setParameters("acc", DecisionTableSParameter.SN5.getInfo(), false, 1);
		dt1.setParameters("acc", DecisionTableSParameter.SN3.getInfo(), false, 1);
		dt3.setParameters("acc", DecisionTableSParameter.SN7.getInfo(), false, 1);
		
		
	}
	
}
