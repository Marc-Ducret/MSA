package edu.usc.thevillagers.serversideagent.gui;

import java.nio.ByteBuffer;
import java.util.List;

import net.minecraft.block.state.IBlockState;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.BufferBuilder;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.client.renderer.vertex.DefaultVertexFormats;
import net.minecraft.client.renderer.vertex.VertexFormat;
import net.minecraft.client.renderer.vertex.VertexFormatElement;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.IBlockAccess;

public class ChunkBufferManager {

	private BufferBuilder[] buffers;
	private boolean[] updateRequired;
	
	private int fromX, fromY, fromZ, toX, toY, toZ; // chunk coordinates
	private int sizeX, sizeY, sizeZ;
	
	public ChunkBufferManager(BlockPos from, BlockPos to) {
		fromX = from.getX() >> 4;
		fromY = from.getY() >> 4;
		fromZ = from.getZ() >> 4;
		toX = to.getX() >> 4;
		toY = to.getY() >> 4;
		toZ = to.getZ() >> 4;
		sizeX = toX - fromX + 1;
		sizeY = toY - fromY + 1;
		sizeZ = toZ - fromZ + 1;
		buffers = new BufferBuilder[sizeX * sizeY * sizeZ];
		updateRequired = new boolean[sizeX * sizeY * sizeZ];
	}
	
	public void renderSubChunk(IBlockAccess blockAccess, int chunkX, int chunkY, int chunkZ, boolean update) {
		int index = subChunkIndex(chunkX, chunkY, chunkZ);
		if(index < 0) return;
		if(update && updateRequired[index]) {
			updateSubChunk(blockAccess, chunkX, chunkY, chunkZ);
			updateRequired[index] = false;
		}
		if(buffers[index] != null) {
			drawBuffer(buffers[index]);
		}
	}
	
	public void requestUpdate(int chunkX, int chunkY, int chunkZ) {
		int index = subChunkIndex(chunkX, chunkY, chunkZ);
		if(index < 0) return;
		updateRequired[index] = true;
	}
	
	private int subChunkIndex(int chunkX, int chunkY, int chunkZ) {
		if(chunkX < fromX || chunkX > toX || chunkY < fromY || chunkY > toY || chunkZ < fromZ || chunkZ > toZ)
			return -1;
		return ((chunkZ - fromZ) * sizeY + (chunkY - fromY)) * sizeX + (chunkX - fromX);
	}
	
	private void updateSubChunk(IBlockAccess blockAccess, int chunkX, int chunkY, int chunkZ) {
		int index = subChunkIndex(chunkX, chunkY, chunkZ);
		if(index < 0) return;
		BufferBuilder buffer = buffers[index];
		if(buffer == null)
			buffer = new BufferBuilder(0x8000);
		buffer.reset();
		buffer.begin(7, DefaultVertexFormats.BLOCK);
		buffer.setTranslation(0, 0, 0);
		for(BlockPos p : BlockPos.getAllInBoxMutable(chunkX << 4, chunkY << 4, chunkZ << 4, 
											(chunkX << 4) + 15, (chunkY << 4) + 15, (chunkZ << 4) + 15)) {
			IBlockState state = blockAccess.getBlockState(p);
			Minecraft.getMinecraft().getBlockRendererDispatcher().renderBlock(state, p, blockAccess, buffer);
		}
		buffer.finishDrawing();
		buffers[index] = buffer;
	}
	
	private void drawBuffer(BufferBuilder buffer) {
		if (buffer.getVertexCount() > 0) {
            VertexFormat vertexformat = buffer.getVertexFormat();
            int i = vertexformat.getNextOffset();
            ByteBuffer bytebuffer = buffer.getByteBuffer();
            List<VertexFormatElement> list = vertexformat.getElements();

            for (int j = 0; j < list.size(); ++j)
            {
                VertexFormatElement vertexformatelement = list.get(j);
                bytebuffer.position(vertexformat.getOffset(j));
                vertexformatelement.getUsage().preDraw(vertexformat, j, i, bytebuffer);
            }

            GlStateManager.glDrawArrays(buffer.getDrawMode(), 0, buffer.getVertexCount());
            int i1 = 0;

            for (int j1 = list.size(); i1 < j1; ++i1)
            {
                VertexFormatElement vertexformatelement1 = list.get(i1);
                vertexformatelement1.getUsage().postDraw(vertexformat, i1, i, bytebuffer);
            }
        }
	}
}
