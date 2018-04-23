package edu.usc.thevillagers.serversideagent.gui;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;

import org.lwjgl.input.Keyboard;
import org.lwjgl.input.Mouse;
import org.lwjgl.opengl.GL11;
import org.lwjgl.util.glu.Project;

import edu.usc.thevillagers.serversideagent.recording.ReplayWorldAccess;
import edu.usc.thevillagers.serversideagent.recording.WorldRecord;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.client.gui.GuiButton;
import net.minecraft.client.gui.GuiScreen;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.client.renderer.OpenGlHelper;
import net.minecraft.client.renderer.RenderHelper;
import net.minecraft.client.renderer.entity.RenderManager;
import net.minecraft.client.renderer.texture.TextureMap;
import net.minecraft.client.renderer.tileentity.TileEntityRendererDispatcher;
import net.minecraft.entity.Entity;
import net.minecraft.tileentity.TileEntity;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraftforge.fml.client.config.GuiSlider;

public class GuiReplay extends GuiScreen {
	
	private int renderDistance = 256;
	
	private WorldRecord record;
	private int speed;
	
	private Vec3d camPos = Vec3d.ZERO.addVector(0, 10, 0), prevCamPos = camPos;
	private float camYaw = 0, prevCamYaw = camYaw;
	private float camPitch = -90, prevCamPitch = camPitch;
	
	private BlockPos renderFrom, renderTo;
	
	private GuiSlider seekSlider;
	
	public GuiReplay(File recordFolder) {
		record = new WorldRecord(recordFolder);
		speed = 1;
	}

	@Override
	public void initGui() {
		super.initGui();
		try {
			record.readInfo();
			
			record.seek(0);
			mc.renderGlobal.setWorldAndLoadRenderers(record.getReplayWorld().fakeWorld);
			for(int chunkZ = record.from.getZ() >> 4; chunkZ < record.to.getZ() >> 4; chunkZ++)
				for(int chunkY = record.from.getY() >> 4; chunkY < record.to.getY() >> 4; chunkY++)
					for(int chunkX = record.from.getX() >> 4; chunkX < record.to.getX() >> 4; chunkX++)
						record.getReplayWorld().chunkBufferManager.requestUpdate(chunkX, chunkY, chunkZ);
			
			camPos = new Vec3d(record.from.add(record.to)).scale(.5);
		} catch(Exception e) {
			e.printStackTrace();
		}
		addButton(new GuiButton(0, 0, height-20, 20, 20, "<"));
		addButton(new GuiButton(1, 50, height-20, 20, 20, ">"));
		seekSlider = new GuiSlider(2, width/2 - 100, height-20, 200, 20, "", " s", 0, record.duration / 20F, 10, false, true) {
			@Override
			public void mouseReleased(int x, int y) {
				super.mouseReleased(x, y);
				mc.addScheduledTask(() -> {
					try {
						record.seek((int) Math.round(getValue() * 20));
					} catch (Exception e) {
						e.printStackTrace();
					}
				});
			}
		};
		addButton(seekSlider);
	}
	
	@Override
	protected void actionPerformed(GuiButton button) throws IOException {
		super.actionPerformed(button);
		switch(button.id) {
		case 0:
			if(speed > 1)
				speed /= 2;
			else
				speed = 0;
			break;
		case 1:
			if(speed < 2048)
				speed *= 2;
			if(speed == 0)
				speed = 1;
			break;
		}
	}
	
	@Override
	protected void mouseClicked(int mouseX, int mouseY, int mouseButton) throws IOException {
		super.mouseClicked(mouseX, mouseY, mouseButton);
		for(GuiButton b : buttonList)
			if(b.isMouseOver())
				return;
		Mouse.setGrabbed(mouseButton == 0);
	}
	
	@Override
	public void onGuiClosed() {
		super.onGuiClosed();
		Mouse.setGrabbed(false);
	}
	
	@Override
	public void updateScreen() {
		super.updateScreen();
		Vec3d move = Vec3d.ZERO;
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindForward.getKeyCode())) move = move.addVector(+0, +0, -1);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindBack	.getKeyCode())) move = move.addVector(+0, +0, +1);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindRight	.getKeyCode())) move = move.addVector(+1, +0, +0);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindLeft	.getKeyCode())) move = move.addVector(-1, +0, +0);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindJump	.getKeyCode())) move = move.addVector(+0, +1, +0);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindSneak	.getKeyCode())) move = move.addVector(+0, -1, +0);
		move = move.normalize();
		prevCamPos = camPos;
		prevCamYaw = camYaw;
		prevCamPitch = camPitch;
		
		if(Mouse.isGrabbed()) {
			camPos = camPos.add(move.rotateYaw((float)Math.toRadians(-camYaw)).scale(.5));
			
			camYaw += Mouse.getDX() * .2;
			camPitch += Mouse.getDY() * .2;
		}
		EntityPlayerSP fakePlayer = record.getReplayWorld().fakePlayer;
		fakePlayer.prevRotationPitch 	= fakePlayer.rotationPitch;
		fakePlayer.prevRotationYaw 		= fakePlayer.rotationYaw;
		fakePlayer.lastTickPosX			= fakePlayer.posX;
		fakePlayer.lastTickPosY 		= fakePlayer.posY;
		fakePlayer.lastTickPosZ 		= fakePlayer.posZ;
		fakePlayer.rotationYaw 			= 180+camYaw;
		fakePlayer.rotationPitch 		= -camPitch;
		fakePlayer.posX					= camPos.x;
		fakePlayer.posY 				= camPos.y;
		fakePlayer.posZ					= camPos.z;
		
		try {
			for(int i = 0; i < speed; i ++)
				record.endReplayTick();
			seekSlider.setValue(record.currentTick / 20F);
			seekSlider.updateSlider();
		} catch (InterruptedException | ExecutionException e) {
			throw new RuntimeException("Replay tick failure", e);
		}
		
		mc.getTextureMapBlocks().tick();
	}
	
	private void setupCamera(float partialTicks) {
		GlStateManager.matrixMode(GL11.GL_PROJECTION);
		GlStateManager.pushMatrix();
        GlStateManager.loadIdentity();
        Project.gluPerspective(90, (float)this.mc.displayWidth / (float)this.mc.displayHeight, 0.05F, 1000 * 2.0F);
        
        GlStateManager.matrixMode(GL11.GL_MODELVIEW);
        GlStateManager.pushMatrix();
		GlStateManager.loadIdentity();
		GlStateManager.color(1.0F, 1.0F, 1.0F, 1.0F);
        float curCamPitch = prevCamPitch + (camPitch - prevCamPitch) * partialTicks;
		float curCamYaw   = prevCamYaw	 + (camYaw   - prevCamYaw  ) * partialTicks;
		GlStateManager.rotate(-curCamPitch, 1, 0, 0);
		GlStateManager.rotate(curCamYaw, 0, 1, 0);
		Vec3d curCamPos = prevCamPos.add(camPos.subtract(prevCamPos).scale(partialTicks));
		GlStateManager.translate(-curCamPos.x, -curCamPos.y, -curCamPos.z);
		
		renderFrom = new BlockPos(	Math.max(record.from.getX(), (int) Math.floor(curCamPos.x) - renderDistance),
									Math.max(record.from.getY(), (int) Math.floor(curCamPos.y) - renderDistance),
									Math.max(record.from.getZ(), (int) Math.floor(curCamPos.z) - renderDistance));
		renderTo = new BlockPos(	Math.min(record.to  .getX(), (int) Math.floor(curCamPos.x) + renderDistance),
									Math.min(record.to  .getY(), (int) Math.floor(curCamPos.y) + renderDistance),
									Math.min(record.to  .getZ(), (int) Math.floor(curCamPos.z) + renderDistance));
	}
	
	private void renderBlocks(ReplayWorldAccess world) {
		GlStateManager.enableDepth();
		GlStateManager.enableCull();
        this.mc.entityRenderer.enableLightmap();
        GlStateManager.enableAlpha();
        GlStateManager.enableBlend();
        GlStateManager.shadeModel(GL11.GL_SMOOTH);
        	
        mc.renderEngine.bindTexture(TextureMap.LOCATION_BLOCKS_TEXTURE);
		
        for(int chunkZ = renderFrom.getZ() >> 4; chunkZ <= renderTo.getZ() >> 4; chunkZ++)
			for(int chunkY = renderFrom.getY() >> 4; chunkY <= renderTo.getY() >> 4; chunkY++)
				for(int chunkX = renderFrom.getX() >> 4; chunkX <= renderTo.getX() >> 4; chunkX++) {
					world.chunkBufferManager.renderSubChunk(world, chunkX, chunkY, chunkZ);
				}
	}
	
	private void renderEntities(ReplayWorldAccess world) { //TODO add vision check
		RenderHelper.enableStandardItemLighting();
		RenderManager renderManager = mc.getRenderManager();
		renderManager.setPlayerViewY(180);
		renderManager.setRenderShadow(false);
		renderManager.renderViewEntity = world.fakePlayer;
		renderManager.setRenderPosition(0, 0, 0);
		AxisAlignedBB renderBounds = new AxisAlignedBB(renderFrom, renderTo);
		for(Entity e : world.getEntities()) {
			if(renderBounds.contains(e.getPositionVector())) {
				this.mc.entityRenderer.disableLightmap();
				renderManager.renderEntityStatic(e, 1, false);
			}
		}
		TileEntityRendererDispatcher.instance.prepare(world.fakeWorld, mc.getTextureManager(), mc.fontRenderer, world.fakePlayer, null, 1);
		TileEntityRendererDispatcher.instance.preDrawBatch();
		GlStateManager.translate(TileEntityRendererDispatcher.staticPlayerX, TileEntityRendererDispatcher.staticPlayerY, TileEntityRendererDispatcher.staticPlayerZ);
		for(TileEntity tileEntity : world.getTileEntities()) {
			if(renderBounds.contains(new Vec3d(tileEntity.getPos()))) {
				this.mc.entityRenderer.disableLightmap();
				TileEntityRendererDispatcher.instance.render(tileEntity, 0, -1);
			}
		}
		TileEntityRendererDispatcher.instance.drawBatch(0);
		renderManager.setRenderShadow(true);
		
		RenderHelper.disableStandardItemLighting();
        GlStateManager.disableRescaleNormal();
        GlStateManager.setActiveTexture(OpenGlHelper.lightmapTexUnit);
        GlStateManager.disableTexture2D();
        GlStateManager.setActiveTexture(OpenGlHelper.defaultTexUnit);
	}
	
	@Override
	public void drawScreen(int mouseX, int mouseY, float partialTicks) {
		drawDefaultBackground();
		
		ReplayWorldAccess world = record.getReplayWorld();
		mc.world = world.fakeWorld;
		mc.player = world.fakePlayer;
		mc.playerController = world.fakePlayerController;
		mc.setRenderViewEntity(mc.player);
		
		try {
			setupCamera(partialTicks);
	        renderBlocks(world);
	        renderEntities(world);
	        GlStateManager.matrixMode(GL11.GL_PROJECTION);
			GlStateManager.popMatrix();
			GlStateManager.matrixMode(GL11.GL_MODELVIEW);
			GlStateManager.popMatrix();
			GlStateManager.disableDepth();
		} catch(Exception e) {
			e.printStackTrace();
		}
		
		mc.world = null;
		mc.player = null;
		mc.playerController = null;
		super.drawScreen(mouseX, mouseY, partialTicks);
		drawCenteredString(mc.fontRenderer, speed+"*", 35, height-14, 0xFFFFFF);
	}
	
	@Override
	public boolean doesGuiPauseGame() {
		return false;
	}
}