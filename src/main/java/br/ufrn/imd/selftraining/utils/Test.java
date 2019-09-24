package br.ufrn.imd.selftraining.utils;

public class Test {

	
	public static void main(String[] args) {
		int size = 2358;
		
		int percent = 10;
		int amount = (int)size / percent;
		
		System.out.println (size + "\n" +
							percent + "\n" +
							amount);
		
		
		
		for(int i = 0; i < 10;i++ ){
			if(size < amount*2) {
				amount = size;
			}
			int control = 0;
			for(int j = 0; j < amount; j++) {
				size--;
				control++;
			}
			System.out.println("adicionados: " + control);
		}
		
	}
	
	
}
