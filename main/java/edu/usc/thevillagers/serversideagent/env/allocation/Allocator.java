package edu.usc.thevillagers.serversideagent.env.allocation;

import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

public interface Allocator {

	/**
	 * Attempts to allocate a space for an environment in the world
	 * @param world
	 * @return @Nullable BlockPos the origin of the allocation
	 */
	public BlockPos allocate(World world);
	
	/**
	 * Frees the allocation that was made at origin
	 * @param world
	 * @param ref
	 */
	public void free(World world, BlockPos origin);
}
