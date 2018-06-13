package edu.usc.thevillagers.serversideagent.env;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.env.actuator.Actuator;
import edu.usc.thevillagers.serversideagent.env.allocation.Allocator;
import edu.usc.thevillagers.serversideagent.env.sensor.Sensor;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

public class EnvironmentTrade extends Environment {
	
	/**
	 * number of citizens
	 */
	private int C;
	
	/**
	 * return rate (reward given to other for one reward consumed)
	 */
	private int R;
	
	/**
	 * time limit
	 */
	private int T;
	
	private StatEntry stat;
	
	@Override
	public void readPars(float[] pars) {
		C = getRoundPar(pars, 0, 10);
		R = getRoundPar(pars, 1, 2);
		T = getRoundPar(pars, 2, 100);
		stat = new StatEntry(C, R, T, 0, 0, 0);
		allocator = new Allocator() {
			
			@Override
			public void free(World world, BlockPos origin) {
			}
			
			@Override
			public BlockPos allocate(World world) {
				return new BlockPos(1337, 128, 1337);
			}
		};
	}

	@Override
	protected void buildSensors() {
		sensors.add(new Sensor(C-1) {
			@Override
			public void sense(Actor actor) {
				Citizen c = (Citizen) actor.envData;
				for(int i = 0; i < C-1; i++) 
					values[i] = c.reputation[i];
			}
		});
		sensors.add(new Sensor(C-1) {
			@Override
			public void sense(Actor actor) {
				Citizen c = (Citizen) actor.envData;
				for(int i = 0; i < C-1; i++) {
					Citizen p = c.getPeer(i);
					values[i] = p.deceive[p.toLocal(c.index)] ? +1 : -1;
				}
			}
		});
	}

	@Override
	protected void buildActuators() {
		actuators.add(new Actuator(C-1) {
			@Override
			public Reverser reverser(Actor actor, WorldRecordReplayer replay) {
				return null;
			}
			
			@Override
			public void act(Actor actor) {
				Citizen c = (Citizen) actor.envData;
				for(int i = 0; i < C-1; i++) 
					c.deceive[i] = values[i] < 0;
			}
		});
		actuators.add(new Actuator(C-1) {
			@Override
			public Reverser reverser(Actor actor, WorldRecordReplayer replay) {
				return null;
			}
			
			@Override
			public void act(Actor actor) {
				Citizen c = (Citizen) actor.envData;
				for(int i = 0; i < C-1; i++) {
					float weight = 5F / T;
					c.reputation[i] = (1-weight) * c.reputation[i] + weight * values[i];
				}
			}
		});
	}

	@Override
	protected void stepActor(Actor a) throws Exception {
		a.reward = 0;
		Citizen c = (Citizen) a.envData;
		for(int i = 0; i < C-1; i++) {
			Citizen p = c.getPeer(i);
			boolean cDp = c.deceive[i], pDc = p.deceive[p.toLocal(c.index)];
			if(!cDp) a.reward -= 1;
			if(!pDc) a.reward += R;
			stat.update(!cDp && !pDc ? 1 : 0, cDp ^ pDc ? 1 : 0, cDp && pDc ? 1 : 0);
		}
		if(time >= T) done = true;
	}
	
	private int index = 0;
	private Citizen[] citizens;
	@Override
	public void reset() {
		super.reset();
		if(!isEmpty()) {
			citizens = new Citizen[C];
			applyToActiveActors((a) -> {
				if(index < C) {
					citizens[index] = new Citizen(index);
					a.envData = citizens[index];
				}
				index++;
			});
			if(index != C) {
				System.out.println("Incorect number of agents ("+index+" instead of "+C+")");
				done = true;
			}
			index = 0;
		}
	}
	
	@Override
	public void terminate() {
		super.terminate();
		stats.add(stat);
	}
	
	private class Citizen {
		int index;
		float[] reputation;
		boolean[] deceive;
		
		Citizen(int index) {
			reputation = new float[C-1];
			deceive = new boolean[C-1];
			this.index = index;
		}
		
		int toLocal(int glob) {
			return Math.floorMod(glob + (C-1) - index, C);
		}
		
		int toGlobal(int loc) {
			return Math.floorMod(loc - (C-1) + index, C);
		}
		
		Citizen getPeer(int loc) {
			return citizens[toGlobal(loc)];
		}
	}
	
	private static class StatEntry {
		int c, r, t;
		float cooperate, deceiveOne, deceiveBoth; 
		
		StatEntry(int c, int r, int t, float cooperate, float deceiveOne, float deceiveBoth) {
			this.c = c;
			this.r = r;
			this.t = t;
			this.cooperate = cooperate;
			this.deceiveOne = deceiveOne;
			this.deceiveBoth = deceiveBoth;
		}
		
		void update(float cooperate, float deceiveOne, float deceiveBoth) {
			this.cooperate = (1 - statHorizon) * this.cooperate + statHorizon * cooperate;
			this.deceiveOne = (1 - statHorizon) * this.deceiveOne + statHorizon * deceiveOne;
			this.deceiveBoth = (1 - statHorizon) * this.deceiveBoth + statHorizon * deceiveBoth;
		}
	}
	
	private static float statHorizon = .001F;
	private static List<StatEntry> stats = new ArrayList<>();
	
	private static String encodeStats() {
		StringBuilder encoding = new StringBuilder();
		encoding.append('{');
		for(StatEntry stat : stats) {
			encoding.append(String.format("(%d, %d, %d): (%f, %f, %f), ", stat.c, stat.r, stat.t, stat.cooperate, stat.deceiveOne, stat.deceiveBoth));
		}
		encoding.append('}');
		return encoding.toString();
	}
	
	public static void writeStats() throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File("stats")));
		writer.write(encodeStats());
		writer.close();
	}
	
	public static void resetStats() throws IOException {
		stats.clear();
	}
}
