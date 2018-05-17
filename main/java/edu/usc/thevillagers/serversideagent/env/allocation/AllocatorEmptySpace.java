package edu.usc.thevillagers.serversideagent.env.allocation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import net.minecraft.init.Blocks;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

/**
 * Allocation in a box that must not overlap any other allocated box and only contain
 * air blocks. Some boxes are randomly sampled to try to allocate one.
 */
public class AllocatorEmptySpace implements Allocator {
	
	private static final int SAMPLES = 30;
	private static final BlockPos SAMPLE_MIN = new BlockPos(-20, 10, -20), SAMPLE_MAX = new BlockPos(20, 200, 20);
	private static final List<AxisAlignedBB> takenSpace = new ArrayList<>();
	
	private final BlockPos min, max;
	private AxisAlignedBB boundingBox;
	
	private final Random rand = new Random();
	
	/**
	 * The bounds of the box required by the environment (relative to its origin)
	 * @param min
	 * @param max
	 */
	public AllocatorEmptySpace(BlockPos min, BlockPos max) {
		this.min = min;
		this.max = max;
		boundingBox = new AxisAlignedBB(min, max);
	}
	
	private BlockPos sample() {
		BlockPos size = SAMPLE_MAX.subtract(SAMPLE_MIN).add(1, 1, 1);
		return new BlockPos(
				SAMPLE_MIN.getX() + rand.nextInt(size.getX()),
				SAMPLE_MIN.getY() + rand.nextInt(size.getY()),
				SAMPLE_MIN.getZ() + rand.nextInt(size.getZ()));
	}
	
	private boolean test(BlockPos origin, World world) {
		for(AxisAlignedBB bb : takenSpace)
			if(bb.intersects(boundingBox.offset(origin)))
				return false;
		for(BlockPos p : BlockPos.getAllInBoxMutable(origin.add(min), origin.add(max)))
			if(world.getBlockState(p).getBlock() != Blocks.AIR) 
				return false;
		return true;
	}
	
	private void reserve(BlockPos origin) {
		boundingBox = boundingBox.offset(origin);
		takenSpace.add(boundingBox);
	}

	@Override
	public BlockPos allocate(World world) {
		for(int i = 0; i < SAMPLES; i ++) {
			BlockPos sample = i > 0 ? sample() : new BlockPos(0, 15, 0);
			if(test(sample, world)) {
				reserve(sample);
				return sample;
			}
		}
		return null;
	}

	@Override
	public void free(World world, BlockPos origin) {
		for(BlockPos p : BlockPos.getAllInBox(origin.add(min), origin.add(max)))
			world.setBlockState(p, Blocks.AIR.getDefaultState());
		takenSpace.remove(boundingBox);
	}
}
