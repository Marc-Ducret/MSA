package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorHit;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorLook;
import edu.usc.thevillagers.serversideagent.env.allocation.AllocatorEmptySpace;
import edu.usc.thevillagers.serversideagent.env.sensor.SensorRaytrace;
import net.minecraft.block.BlockColored;
import net.minecraft.block.state.IBlockState;
import net.minecraft.init.Blocks;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.RayTraceResult;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;

public class EnvironmentBreakGold extends Environment {
	
	private int size;
	private BlockPos goldBlock;
	
	@Override
	public void readPars(float[] pars) {
		size = getRoundPar(pars, 0, 2);
		allocator = new AllocatorEmptySpace(new BlockPos(-size, -1, -size), new BlockPos(size, 2, size));
	}
	
	@Override
	protected void buildSensors() {
		sensors.add(new SensorRaytrace(12, 6, 1, 70, 2) {
			
			@Override
			protected void encode(World world, Vec3d from, Vec3d dir, RayTraceResult res, float[] result) {
				if(res == null) {
					result[0] = -1;
					return;
				}
				IBlockState state = world.getBlockState(res.getBlockPos());
				if(state.getBlock() == Blocks.STAINED_GLASS) {
					result[0] = state.getValue(BlockColored.COLOR) == EnumDyeColor.YELLOW ? 1 : 0;
				} else {
					result[0] = -1;
				}
			}
		});
	}
	
	@Override
	protected void buildActuators() {
		actuators.add(new ActuatorHit());
		actuators.add(new ActuatorLook());
	}
	
	@Override
	public void reset() {
		super.reset();
		generate();
	}
	
	private void generate() {
		BlockPos ref = getOrigin();
		for(int z =- size; z <= size; z++) {
			for(int x =- size; x <= size; x++) {
				world.setBlockState(ref.add(x, -1, z), 
						Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.LIME));
			}
		}
		if(goldBlock != null) world.setBlockState(goldBlock, Blocks.AIR.getDefaultState());
		int x = world.rand.nextInt(2 * size + 1) - size;
		int z = world.rand.nextInt(2 * size + 1) - size;
		goldBlock = ref.add(x, 0, z);
		world.setBlockState(goldBlock, 
				Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.YELLOW));
	}

	@Override
	protected void stepActor(Actor a) throws Exception {
		BlockPos p = a.entity.getPosition().down();
		IBlockState state = world.getBlockState(p);
		a.reward = -.01F;
		if(world.getBlockState(goldBlock).getBlock() != Blocks.STAINED_GLASS) {
			done = true;
			a.reward = 10;
		}
		if(state.getBlock() == Blocks.STAINED_GLASS) {
			if(time > 99) {
				done = true;
				a.reward = -1;
			}
		} else {
			done = true;
			a.reward = -10;
		}
	}
	
	@Override
	protected void step() {
		super.step();
	}
}
