package edu.usc.thevillagers.serversideagent.env;

import java.util.ArrayList;
import java.util.List;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorForwardStrafe;
import edu.usc.thevillagers.serversideagent.env.allocation.AllocatorEmptySpace;
import edu.usc.thevillagers.serversideagent.env.sensor.Sensor;
import net.minecraft.block.BlockColored;
import net.minecraft.init.Blocks;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.util.math.BlockPos;

public class EnvironmentLandmarks extends Environment {
	
	private int nLandmarks;
	private int nAgents;
	private int size;
	private int comSize;
	
	private float[][] comChanels;
	
	private List<Actor> actors;
	private BlockPos[] landmarks;
	
	
	@Override
	public void readPars(float[] pars) {
		nLandmarks = getRoundPar(pars, 0, 4);
		nAgents = getRoundPar(pars, 1, 4);
		comSize = getRoundPar(pars, 2, 1);
		size = getRoundPar(pars, 3, 8);
		comChanels = new float[nAgents][];
		for(int i = 0; i < comSize; i ++) 
			comChanels[i] = new float[comSize];
		actors = new ArrayList<>(nAgents);
		landmarks = new BlockPos[nLandmarks];
		allocator = new AllocatorEmptySpace(new BlockPos(-size, -1, -size), new BlockPos(size, 2, size));
	}
	
	@Override
	protected void buildSensors() {
		sensors.add(new Sensor((nLandmarks + nAgents) * 2 + nAgents * comSize) {
			
			@Override
			public void sense(Agent agent) {
				BlockPos p = agent.entity.getPosition();
				int offset = actors.indexOf(agent);
				for(int i = 0; i < nAgents; i ++) {
					int iAgent = (i + offset) % nAgents;
					if(actors.size() > iAgent) {
						BlockPos aP = actors.get(iAgent).entity.getPosition();
						values[i * 2 + 0] = aP.getX() - p.getX();
						values[i * 2 + 1] = aP.getZ() - p.getZ();
					}
				}
				for(int i = 0; i < nLandmarks; i++) {
					BlockPos lP = landmarks[i];
					values[2 * nAgents + i * 2 + 0] = lP.getX() - p.getX();
					values[2 * nAgents + i * 2 + 1] = lP.getZ() - p.getZ();
				}
				for(int i = 0; i < 2 * (nAgents + nLandmarks); i++) values[i] /= size;
				for(int i = 0; i < nAgents; i ++)
					for(int c = 0; c < comSize; c++)
						values[2 * (nAgents + nLandmarks) + i * comSize + c] = comChanels[(i + offset) % nAgents][c];
			}
		});
	}
	
	@Override
	protected void buildActuators() {
		actuators.add(new ActuatorForwardStrafe());
	}
	
	@Override
	public void reset() {
		super.reset();
		actors.clear();
		applyToActiveActors((a) -> {
			if(actors.size() < nAgents) actors.add(a);
		});
		for(int i = 0; i < nLandmarks; i++) {
			landmarks[i] = getOrigin().add(world.rand.nextInt(2*size - 1) - size + 1, -1, world.rand.nextInt(2*size - 1) - size + 1);
		}
		generate();
	}
	
	private void generate() {
		BlockPos ref = getOrigin();
		for(int z =- size; z <= size; z++) {
			for(int x =- size; x <= size; x++) {
				world.setBlockState(ref.add(x, -1, z), Blocks.STAINED_GLASS.getDefaultState());
				for(int y = 0; y < 2; y ++)
					world.setBlockState(ref.add(x, y, z), 
							Math.max(Math.abs(x), Math.abs(z)) == size ? 
										Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.BLACK) :
										Blocks.AIR.getDefaultState());
			}
		}
		for(BlockPos p : landmarks) {
			world.setBlockState(p, Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.BLUE));
		}
	}

	@Override
	protected void stepActor(Actor a) throws Exception {
		
	}
	
	@Override
	protected void step() {
		super.step();
		float reward = 0;
		for(BlockPos landmark : landmarks) {
			float distSq = size*size;
			for(Actor a : actors) distSq = Math.min(distSq, (float) landmark.distanceSq(a.entity.getPosition()));
			reward -= distSq * .1F;
		}
		for(Actor a : actors)
			a.reward = reward / nAgents;
		if(time >= 49)
			done = true;
	}

}
