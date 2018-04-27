package edu.usc.thevillagers.serversideagent.gui;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.concurrent.ExecutionException;

import org.lwjgl.input.Keyboard;
import org.lwjgl.input.Mouse;
import org.lwjgl.opengl.GL11;
import org.lwjgl.util.glu.Project;

import edu.usc.thevillagers.serversideagent.recording.ReplayWorldAccessClient;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayerClient;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.client.gui.GuiButton;
import net.minecraft.client.gui.GuiScreen;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.client.renderer.OpenGlHelper;
import net.minecraft.client.renderer.RenderHelper;
import net.minecraft.client.renderer.entity.RenderManager;
import net.minecraft.client.renderer.texture.TextureMap;
import net.minecraft.client.renderer.tileentity.TileEntityRendererDispatcher;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityLivingBase;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.init.Items;
import net.minecraft.item.ItemStack;
import net.minecraft.tileentity.TileEntity;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraftforge.fml.client.config.GuiSlider;

public class GuiReplay extends GuiScreen {
	
	private int renderDistance = 128;
	
	private WorldRecordReplayerClient record;
	private int speed;
	
	private Vec3d camPos = Vec3d.ZERO.addVector(0, 10, 0), prevCamPos = camPos;
	private float camYaw = 0, prevCamYaw = camYaw;
	private float camPitch = -90, prevCamPitch = camPitch;
	
	private Entity followedEntity;
	
	private BlockPos renderFrom, renderTo;
	
	private GuiSlider seekSlider;
	
	public GuiReplay(File recordFolder) {
		record = new WorldRecordReplayerClient(recordFolder);
		speed = 1;
	}

	@Override
	public void initGui() {
		super.initGui();
		try {
			if(record.world == null) {
				record.readInfo();
				
				record.seek(0);
				
				camPos = new Vec3d(record.from.add(record.to)).scale(.5);
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
		addButton(new GuiButton(0, 0, height-20, 20, 20, "<"));
		addButton(new GuiButton(1, 50, height-20, 20, 20, ">"));
		addButton(new GuiButton(2, 75, height-20, 20, 20, "+1"));
		seekSlider = new GuiSlider(3, width/2 - 100, height-20, 200, 20, "", " s", 0, record.duration / 20F, 10, true, true) {
			@Override
			public void mouseReleased(int x, int y) {
				super.mouseReleased(x, y);
				mc.addScheduledTask(() -> {
					try {
						followedEntity = null;
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
		case 2:
			try {
				record.endReplayTick();
			} catch (Exception e) {
				throw new RuntimeException("Replay tick failure", e);
			}
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
	protected void keyTyped(char typedChar, int keyCode) throws IOException {
		super.keyTyped(typedChar, keyCode);
		if(keyCode == mc.gameSettings.keyBindInventory.getKeyCode()) {
			if(followedEntity == null) {
				Vec3d rayStart = record.world.fakePlayer.getPositionVector();
				Vec3d rayEnd = rayStart.add(record.world.fakePlayer.getLookVec().scale(5));
				for(Entity e : record.world.getEntities()) {
					if(e.getEntityBoundingBox() != null && e.getEntityBoundingBox().calculateIntercept(rayStart, rayEnd) != null)
						followedEntity = e;
				}
			} else {
				followedEntity = null;
			}
		}
	}
	
	@Override
	public void onGuiClosed() {
		super.onGuiClosed();
		Mouse.setGrabbed(false);
	}
	
	@Override
	public void updateScreen() {
		super.updateScreen();
		prevCamPos = camPos;
		prevCamYaw = camYaw;
		prevCamPitch = camPitch;
		if(followedEntity != null) {
			camPos = followedEntity.getPositionVector().addVector(0, followedEntity.getEyeHeight(), 0);
			camPitch = -followedEntity.rotationPitch;
			camYaw = followedEntity.rotationYaw + 180;
		} else {
			Vec3d move = Vec3d.ZERO;
			if(Keyboard.isKeyDown(mc.gameSettings.keyBindForward.getKeyCode())) move = move.addVector(+0, +0, -1);
			if(Keyboard.isKeyDown(mc.gameSettings.keyBindBack	.getKeyCode())) move = move.addVector(+0, +0, +1);
			if(Keyboard.isKeyDown(mc.gameSettings.keyBindRight	.getKeyCode())) move = move.addVector(+1, +0, +0);
			if(Keyboard.isKeyDown(mc.gameSettings.keyBindLeft	.getKeyCode())) move = move.addVector(-1, +0, +0);
			if(Keyboard.isKeyDown(mc.gameSettings.keyBindJump	.getKeyCode())) move = move.addVector(+0, +1, +0);
			if(Keyboard.isKeyDown(mc.gameSettings.keyBindSneak	.getKeyCode())) move = move.addVector(+0, -1, +0);
			move = move.normalize();
			
			if(Mouse.isGrabbed()) {
				camPos = camPos.add(move.rotateYaw((float)Math.toRadians(-camYaw)).scale(.5));
				
				camYaw += Mouse.getDX() * .2;
				camPitch += Mouse.getDY() * .2;
			}
		}
		
		EntityPlayerSP fakePlayer = record.world.fakePlayer;
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
			for(int i = 0; i < speed; i ++) {
				record.endReplayTick();
				for(Entity e : record.world.getEntities()) {
					if(e instanceof EntityPlayer) {
						EntityPlayer player = (EntityPlayer) e;
						try {
							Method method = EntityLivingBase.class.getDeclaredMethod("updateArmSwingProgress");
							method.setAccessible(true);
							method.invoke(player);
						} catch(Exception ex) {
							ex.printStackTrace();
						}
					}
				}
			}
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
	
	private void renderBlocks(ReplayWorldAccessClient world) {
		GlStateManager.enableDepth();
		GlStateManager.enableCull();
        this.mc.entityRenderer.enableLightmap();
        GlStateManager.enableAlpha();
        GlStateManager.enableBlend();
        GlStateManager.shadeModel(GL11.GL_SMOOTH);
        	
        mc.renderEngine.bindTexture(TextureMap.LOCATION_BLOCKS_TEXTURE);
		long startTime = System.currentTimeMillis();
        
        for(int chunkZ = renderFrom.getZ() >> 4; chunkZ <= renderTo.getZ() >> 4; chunkZ++)
			for(int chunkY = renderFrom.getY() >> 4; chunkY <= renderTo.getY() >> 4; chunkY++)
				for(int chunkX = renderFrom.getX() >> 4; chunkX <= renderTo.getX() >> 4; chunkX++) {
					boolean update = System.currentTimeMillis() - startTime < 100;
					if(!update) {
						int dx = chunkX - (((int)camPos.x) >> 4);
						int dy = chunkY - (((int)camPos.y) >> 4);
						int dz = chunkZ - (((int)camPos.z) >> 4);
						if(dx*dx + dy*dy + dz*dz <= 5*5) update = true;
					}
					world.chunkBufferManager.renderSubChunk(world, chunkX, chunkY, chunkZ, update);
				}
	}
	
	private void renderEntities(ReplayWorldAccessClient world) { //TODO add vision check
		RenderHelper.enableStandardItemLighting();
		RenderManager renderManager = mc.getRenderManager();
		renderManager.setPlayerViewY(180);
		renderManager.setRenderShadow(false);
		renderManager.renderViewEntity = world.fakePlayer;
		renderManager.setRenderPosition(0, 0, 0);
		AxisAlignedBB renderBounds = new AxisAlignedBB(renderFrom, renderTo);
		for(Entity e : world.getEntities()) {
			if(e == followedEntity) continue;
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
				TileEntityRendererDispatcher.instance.render(tileEntity, 1, -1);
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
		
		ReplayWorldAccessClient world = record.world;
		mc.world = (WorldClient) world.fakeWorld;
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
		super.drawScreen(mouseX, mouseY, partialTicks);
		drawCenteredString(mc.fontRenderer, speed+"*", 35, height-14, 0xFFFFFF);
		if(followedEntity != null) {
			drawCenteredString(mc.fontRenderer, followedEntity.getName(), width / 2, 2, 0xFFFFFF);
		}
        itemRender.renderItemAndEffectIntoGUI(new ItemStack(Items.CLOCK), width-18, height-18);
		mc.world = null;
		mc.player = null;
		mc.playerController = null;
	}
	
	@Override
	public boolean doesGuiPauseGame() {
		return false;
	}
}