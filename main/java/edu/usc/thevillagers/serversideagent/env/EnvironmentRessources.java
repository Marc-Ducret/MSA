package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.env.actuator.Actuator;
import edu.usc.thevillagers.serversideagent.env.allocation.Allocator;
import edu.usc.thevillagers.serversideagent.env.sensor.Sensor;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

public class EnvironmentRessources extends Environment {

	private int nRes;
	
	@Override
	public void readPars(float[] pars) {
		nRes = getRoundPar(pars, 0, 3);
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
		sensors.add(new Sensor(nRes) {
			@Override
			public void sense(Actor actor) {
				Citizen c = (Citizen) actor.envData;
				for(int i = 0; i < nRes; i++) values[i] = c.currentRes[i];
			}
		});
		sensors.add(new Sensor(nRes) {
			@Override
			public void sense(Actor actor) {
				Citizen c = (Citizen) actor.envData;
				for(int i = 0; i < nRes; i++) values[i] = c.goalRes[i];
			}
		});
	}

	@Override
	protected void buildActuators() {
		actuators.add(new Actuator(nRes) {
			@Override
			public Reverser reverser(Actor actor, WorldRecordReplayer replay) {
				return null;
			}
			
			@Override
			public void act(Actor actor) {
				Citizen c = (Citizen) actor.envData;
				for(int i = 0; i < nRes; i++) 
					if(values[i] < values[c.workRes])
						c.workRes = i;
			}
		});
	}

	@Override
	protected void stepActor(Actor a) throws Exception {
		a.reward = 0;
		Citizen c = (Citizen) a.envData;
		c.currentRes[c.workRes]++;
		if(c.consumeGoal()) {
			a.reward += 1;
			c.newGoal();
		}
		if(time >= 10) done = true;
	}
	
	@Override
	public void resetActor(Actor a) {
		super.resetActor(a);
		a.envData = new Citizen();
		((Citizen)a.envData).newGoal();
	}
	
	private class Citizen {
		int[] currentRes;
		int[] goalRes;
		int workRes;
		
		Citizen() {
			currentRes = new int[nRes];
			goalRes = new int[nRes];
			workRes = 0;
		}
		
		boolean consumeGoal() {
			boolean reachedGoal = true;
			for(int i = 0; i < nRes; i++)
				if(currentRes[i] < goalRes[i])
					reachedGoal = false;
			if(reachedGoal)
				for(int i = 0; i < nRes; i++)
					currentRes[i] -= goalRes[i];
			return reachedGoal;
		}
		
		void newGoal() {
			for(int i = 0; i < nRes; i++)
				goalRes[i] = 0;
			goalRes[world.rand.nextInt(nRes)] = 3;
		}
	}
}
