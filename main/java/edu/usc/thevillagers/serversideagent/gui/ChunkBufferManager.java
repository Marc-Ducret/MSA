package edu.usc.thevillagers.serversideagent.gui;

import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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

	private Map<Long, BufferBuilder> buffers = new HashMap<>();
	private Set<Long> updatesRequired = new HashSet<>();
	
	public void renderSubChunk(IBlockAccess blockAccess, int chunkX, int chunkY, int chunkZ) {
		long index = subChunkIndex(chunkX, chunkY, chunkZ);
		if(updatesRequired.contains(index)) {
			updateSubChunk(blockAccess, chunkX, chunkY, chunkZ);
			updatesRequired.remove(index);
		}
		if(buffers.containsKey(index)) {
			drawBuffer(buffers.get(index));
		}
	}
	
	public void requestUpdate(int chunkX, int chunkY, int chunkZ) {
		updatesRequired.add(subChunkIndex(chunkX, chunkY, chunkZ));
	}
	
	private long subChunkIndex(int chunkX, int chunkY, int chunkZ) {
		chunkX &= 0xFFFF;
		chunkY &= 0xFF;
		chunkZ &= 0xFFFF;
		return ((long)chunkZ << 32) + ((long)chunkY << 16) + ((long)chunkX); 
	}
	
	private void updateSubChunk(IBlockAccess blockAccess, int chunkX, int chunkY, int chunkZ) {
		long index = subChunkIndex(chunkX, chunkY, chunkZ);
		BufferBuilder buffer = buffers.get(index);
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
		buffers.put(index, buffer);
	}
	
	private void drawBuffer(BufferBuilder buffer) {
		if (buffer.getVertexCount() > 0)
        {
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
