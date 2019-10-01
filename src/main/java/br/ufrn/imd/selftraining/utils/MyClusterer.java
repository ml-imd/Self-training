package br.ufrn.imd.selftraining.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import experimento.weka.naosupervisionados.Mathematics;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.Clusterer;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public abstract class MyClusterer {
	protected Instances base;
	protected Instances baseNotClass;
	protected List<Group> groups;
	protected AbstractClusterer clusterer;
	protected String csvString;
	protected int numGroups;
	
	protected abstract void preProcess(int seed, int nGroups);
	
	protected abstract void posProcess();
	
	public MyClusterer() {
		csvString = this.getCSVHearder()+"\n";
	}
	
	public List<Group> getGroups() {
		return this.groups;
	}
	
	public void init(int numGroups) {
		
		this.groups = new ArrayList<Group>();
		for(int i=0;i<numGroups;i++) {
			groups.add(new Group());
		}
	}
	
	public Clusterer getClusterer() {
		return clusterer;
	}
	
	public void setClusterer(AbstractClusterer clusterer) {
		this.clusterer = clusterer;
	}
	
	public String getCSVHearder() {
		return "MODEL,K,SEED,SILHOUETTE,DB,CR"; 
	}
	
	private void addCSVString(int seed) {
		this.csvString+=this.getClass().getSimpleName()+","+
						this.groups.size()+","+
						seed+","+
						this.getS()+","+
						this.getDB()+","+
						this.getCR()+"\n";
	}
	
	public void toSaveCSVFile(int seed,String filename) {
		
		this.addCSVString(seed);
		
		File file = new File(filename);
		
		if(file.getParentFile()!=null) {
			file.getParentFile().mkdirs();
		}
		
		try {
			Writer writer = new FileWriter(file);
			
			writer.write(csvString);
					
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
	
	public void executeClustering(int nGroups,int seed,Instances base, Instances baseNotClass) throws Exception {
		
		//COMENTE SE NAO QUISER QUE OS DADOS SEJAM NORMALIZADOS
		//----------------------------------------------------
		Filter filterNorm = new Normalize();
        filterNorm.setInputFormat(base);
        base = Filter.useFilter(base, filterNorm);
        
        filterNorm.setInputFormat(baseNotClass);
        baseNotClass = Filter.useFilter(baseNotClass, filterNorm);
        //----------------------------------------------------
        this.numGroups=nGroups;
		this.base =  base;
		this.baseNotClass = baseNotClass;
		
		this.init(nGroups);
		
		this.preProcess(seed,nGroups);
		
		this.clusterer.buildClusterer(baseNotClass);
		
		this.generateStruture(baseNotClass);
		
		this.posProcess();
	}

	private void generateStruture(List<Instance> result) {
		try {
			
			for (Instance i : result) {
				this.groups.get(this.clusterer.clusterInstance(i)).getInstances().add(i);
			}

			//Verificando se o grupo é vazio, caso sim, ele tem que ser retirado
			int cont = 0;
			Iterator<Group> i = this.groups.iterator();
		    		
			while (i.hasNext()) {
				 Group g = (Group) i.next();
		         if (g.getInstances().isEmpty()) {
		            i.remove();
		            System.out.println("WARNING : QUANTIDADE DE GRUPOS MENOR QUE K - O GRUPO " + cont + " EST�? VAZIO - RECOMEND�?VEL MUDAR A SEED QUANDO ISSO ACONTECER" );
		         }
		         cont++;
		      }

			//Obter centróides
			for(Group g : this.groups) {
				g.setCentroid(this.getCentroid(g));
			}
			
		} catch (RuntimeException e) {
			e.printStackTrace();
			System.exit(-1);
		} catch (Exception e) {
			e.printStackTrace();
		}
	
	}

	public double getCR(){
		
		return this.getARI(this.base,this.baseNotClass);
	}
	
	double getARI(Instances X, List<Instance> Y) {
		
		// Tabela de contigencia
		int[][] table = new int[X.numClasses() + 1][this.numGroups + 1];

		// Numero de objetos
		int numInstances = X.numInstances();

		// Popula a tabela
		int xClass, yClass;
		for (int i = 0; i < numInstances; i++) {
			try {
				xClass = (int) X.instance(i).classValue();
				yClass = (int) this.clusterer.clusterInstance(Y.get(i));
	
				table[xClass][yClass]++;
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < table.length - 1; i++) {
			for (int j = 0; j < table[i].length - 1; j++) {
				table[table.length - 1][j] += table[i][j]; // Computa a Ãºltima
															// linha
				table[i][table[i].length - 1] += table[i][j]; // Computa a
																// Ãºltima coluna
			}
		}
	
		double TERMO_A = 0;
		for (int i = 0; i < table.length - 1; i++) {
			for (int j = 0; j < table[i].length - 1; j++) {	
				TERMO_A += Mathematics.combinationOf(table[i][j], 2);
			}
		}
		
		double TERMO_B = 0;
		double TERMO_C = 0;
		for (int i = 0; i < table.length - 1; i++) {
			TERMO_B += Mathematics.combinationOf(
					table[i][table[i].length - 1], 2); // Ultima coluna
			// System.out.printf("B ~> %d-%d\n", i, table[i].length - 1);
		}

		for (int i = 0; i < table[table.length-1].length - 1; i++) {
			TERMO_C += Mathematics.combinationOf(table[table.length - 1][i],
					2); // Ultima linha
			// System.out.printf("C ~> %d-%d\n", table.length - 1, i);
		}

		double TERMO_D = Mathematics.combinationOf(numInstances, 2);

		double INDEX = TERMO_A;
		double EXP_INDEX = (TERMO_B * TERMO_C) / TERMO_D;
		double MAX_INDEX = 0.5 * (TERMO_B + TERMO_C);

//		print(table);

		return ((INDEX - EXP_INDEX) / (MAX_INDEX - EXP_INDEX));
	}
	
	//Euclidian Distance
	public static double distance(Instance x, Instance y) {
		
		if(x==null || y==null) {
			return Double.POSITIVE_INFINITY;
		}
		
		if (x.numAttributes() != y.numAttributes())
	        throw new IllegalArgumentException(String.format("Arrays have different length: x[%d], y[%d]", x.numAttributes(), y.numAttributes()));

	    int n = x.numAttributes();
	    double dist = 0.0;
	    for (int i = 0; i < n; i++) {
	    	if (!Double.isNaN(x.value(i)) && !Double.isNaN(y.value(i))) {
	            double d = x.value(i) - y.value(i);
	            dist += d * d;
	        }
	    }	    
	    return Math.sqrt(dist);
	}
	
	public double getS() {
		
		//s(DATA) = média aritmática de todos os s(G)
		Map<Group,Double> gs = new HashMap<Group,Double>();
		
		for(Group g : this.groups) {
			//s(G) = média aritmética de s(i) 
			Map<Instance,Double> s = new HashMap<Instance,Double>();
			
			for(Instance i : g.getInstances()) {
				//s(i) = valor de silhouette para a instância
				if(g.size()==1) {
					s.put(i,0.0);
				}else {
					double ai=0,bi=0;
					for(Instance j : g.getInstances()) {
						if(!i.equals(j)) {
							double ed = MyClusterer.distance(i, j);
							ai+=ed;
						}	
					}
					
					ai=ai/(g.size() - 1);
					
					bi=Double.MAX_VALUE;
					
					for(Group m : this.groups) {
						double cont=0;
						if(!g.equals(m)) {
							for(Instance j : m.getInstances()) {
								cont+=MyClusterer.distance(i, j);
							}
							cont = cont/m.size();
							
							if(cont<bi) {
								bi = cont;
							}
						}
					}
					//Calcular s(i)
					double result=0;
					
					if(ai<bi) {
						result = 1 - (ai/bi);
					}else {
						if(ai>bi) {
							result = (bi/ai) - 1;
						}
					}
					
					s.put(i,(result));
				}
			}
			
			//CALCULAR MÉDIA DE s(i)
			double cont = 0;
			
			for(Double d : s.values()) {
				cont+=(d/s.size());
			}
			
			gs.put(g,cont);
		}
		
		//CALCULAR MÉDIA DE s(G)
		double cont = 0;
		
		for(Double d : gs.values()) {
			cont+=(d/gs.size());
		}
		
		return cont;
	}	
	
	private Instance getCentroid(Group g) throws RuntimeException{
		
		switch(g.size()) {
			case 0 :
				throw new RuntimeException("GRUPO VAZIO NÃO TEM CENTRÓIDE");
			case 1:
				return g.getInstances().get(0);
			default:
				//Calcular ponto médio e depois escolher a instância mais próxima do ponto médio
				Instance copy = (Instance)g.getInstances().get(0).copy();
				
				for(int idAtt=0;idAtt<copy.numAttributes();idAtt++) {
					double cont=0;
					for(Instance i : g.getInstances()){
						cont+=i.value(idAtt);
					}
					cont=cont/g.size();
					copy.setValue(idAtt, cont);
				}
				
				Instance min = g.getInstances().get(0);
				
				double minValue = MyClusterer.distance(copy, min);
				for(Instance i : g.getInstances()){
					double distance = MyClusterer.distance(copy, i);
					if(distance < minValue) {
						min = i;
						minValue = distance;
					}
				}
				
				return min;
		}
		
	}

	public double getDB() {
		
		//Calcular todos os Si
		Map<Group,Double> S = new HashMap<Group,Double>();
		
		for(Group i : this.groups) {
			double cont=0;
			for(Instance j : i.getInstances()) {
				cont+=(MyClusterer.distance(i.getCentroid(), j)/i.size());
			}
			S.put(i,cont);
		}
		
		//Calcular todos os Mij
		Map<Group, Map<Group, Double>> Mij = new HashMap<Group, Map<Group,Double>>();
		
		for(Group i : this.groups) {
			Mij.put(i, new HashMap<Group, Double>());
			for(Group j : this.groups) {
				//Calculando distancia entre centróides
				Mij.get(i).put(j, MyClusterer.distance(i.getCentroid(), j.getCentroid()));
			}
		}
		//Calcular todos os Rij
		Map<Group, Map<Group, Double>> Rij = new HashMap<Group, Map<Group,Double>>();
		
		for(Group i : this.groups) {
			Rij.put(i, new HashMap<Group, Double>());
			for(Group j : this.groups) {
				Rij.get(i).put(j, (S.get(i) + S.get(j)) / Mij.get(i).get(j) );
			}
		}
		
		//Calcular Di
		Map<Group,Double> Di = new HashMap<Group,Double>();
		
		for(Group i : this.groups) {
			double max = Double.MIN_VALUE;
			for(Group j : this.groups) {
				if(!i.equals(j)) {
					double value = Rij.get(i).get(j);
					if(value > max) {
						max = value;
					}
				}
			}
			Di.put(i, max);
		}
		
		//Calcular Média dos valores de Di
		double cont=0;
		
		for(Double d : Di.values()) {
			cont+=(d/Di.size());
		}
		
		return cont;
	}
}