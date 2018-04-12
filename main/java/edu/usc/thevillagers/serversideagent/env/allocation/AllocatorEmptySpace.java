package edu.usc.thevillagers.serversideagent.env.allocation;

import java.util.Random;

import net.minecraft.init.Blocks;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

public class AllocatorEmptySpace implements Allocator {
	
	private static final int SAMPLES = 30;
	private static final BlockPos SAMPLE_MIN = new BlockPos(-0, 10, -0), SAMPLE_MAX = new BlockPos(0, 200, 0);
	
	private final BlockPos min, max;
	
	private final Random rand = new Random();
	
	/**
	 * The bounds of the box required by the environment (relative to its origin)
	 * @param min
	 * @param max
	 */
	public AllocatorEmptySpace(BlockPos min, BlockPos max) {
		this.min = min;
		this.max = max;
	}
	
	private BlockPos sample() {
		BlockPos size = SAMPLE_MAX.subtract(SAMPLE_MIN).add(1, 1, 1);
		return new BlockPos(
				SAMPLE_MIN.getX() + rand.nextInt(size.getX()),
				SAMPLE_MIN.getY() + rand.nextInt(size.getY()),
				SAMPLE_MIN.getZ() + rand.nextInt(size.getZ()));
	}
	
	private boolean test(BlockPos origin, World world) {
		for(BlockPos p : BlockPos.getAllInBox(origin.add(min), origin.add(max)))
			if(world.getBlockState(p).getBlock() != Blocks.AIR) 
				return false;
		return true;
	}

	@Override
	public BlockPos allocate(World world) {
		for(int i = 0; i < SAMPLES; i ++) {
			BlockPos sample = sample();
			if(test(sample, world))
				return sample;
		}
		return null;
	}

	@Override
	public void free(World world, BlockPos origin) {
		for(BlockPos p : BlockPos.getAllInBox(origin.add(min), origin.add(max)))
			world.setBlockState(p, Blocks.AIR.getDefaultState());
	}
}
